"""
kagome_cnn.py -- size-agnostic, fully convolutional CNN for the kagome Hubbard model.

Lattice -> tensor (L, L, 3): L x L unit cells, 3 sublattices folded into channels.
Every NN bond fits in a 3x3 kernel, so up/down triangles stay distinguishable
(unlike a message-passing GNN). No dense layers on the grid -> works at any L.

Pipeline:
  1. load ED/DMRG samples from .h5 files
  2. print baselines (L0 / L1 / L2) that the CNN must beat
  3. train (batches grouped by lattice size), logging step time and gradient norm
  4. print scaling / convergence summary, physics checks, save plots + model

Requires: numpy, torch, h5py, matplotlib.   Run: python kagome_cnn.py
Assumes square clusters (Lx == Ly), as in your data.
"""

import os
import time
from functools import lru_cache

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONFIG
# =============================================================================
# (folder, number of sites, number of files).  Files are named <prefix>_NNN.h5
TRAIN_SOURCES = [
    ("data_outputs/Tenpy_Outputs1", 12, 90),
    ("data_outputs/Tenpy_Outputs4", 27, 1),
]
TEST_SOURCES = [
    ("data_outputs/Tenpy_Outputs4", 27, 1),
]
SIZE_PREFIX = {
    12: "twelve_site_kagome_",
    27: "twentyseven_site_kagome_",
    48: "fortyeight_site_kagome_",
    75: "seventyfive_site_kagome_",
}

EPOCHS = 100
BATCH_SIZE = 1             # batches never mix lattice sizes
LR = 1e-4
GRAD_CLIP = None           # e.g. 1.0; None = measure the norm but never clip
WIDTH, DEPTH = 64, 4
PRINT_EVERY = 5            # epochs between training log lines
EVAL_EVERY = 10            # epochs between test evaluations
WARMUP_STEPS = 5           # steps excluded from timing statistics
FALLBACK_TARGET_MSE = 1e-3 # convergence target when no L2 baseline exists
CONV_FIT_STEPS = 50        # early window for the loss-decay-rate fit

# Input scaling (U up to ~12 would otherwise dominate the other features)
NORMALIZE_INPUTS = True
FEATURE_SCALE = (12.0, 1.0, 0.5, 1.0)     # U, filling, t1, t2

CHECKPOINT_PATH = "NN_Models/CNN_Model_4.pt"
LOAD_CHECKPOINT = False    # keep False if the checkpoint was trained on unnormalized inputs
SAVE_PATH = "NN_Models/CNN_Model_latest.pt"
PLOT_PATH = "training_diagnostics.png"
SHOW_PLOTS = False
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. KAGOME GEOMETRY
# =============================================================================
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])
BASIS = {0: np.array([0.0, 0.0]), 1: A1 / 2, 2: A2 / 2}

# Directed NN bonds: (d_row, d_col, sublattice_from, sublattice_to)
BOND_TABLE = [
    (-1, 0, 0, 2), (-1, +1, 1, 2),
    (0, -1, 0, 1), (0, 0, 0, 1), (0, 0, 0, 2),
    (0, 0, 1, 0), (0, 0, 1, 2), (0, 0, 2, 0), (0, 0, 2, 1),
    (0, +1, 1, 0), (+1, -1, 2, 1), (+1, 0, 2, 0),
]


def site_positions(L):
    """Cartesian positions, shape (L, L, 3, 2). Flat index = (r*L + c)*3 + s."""
    pos = np.zeros((L, L, 3, 2))
    for r in range(L):
        for c in range(L):
            for s in range(3):
                pos[r, c, s] = r * A2 + c * A1 + BASIS[s]
    return pos


def flat_index(r, c, s, L):
    return (r * L + c) * 3 + s


def build_adjacency(L):
    """Periodic nearest-neighbour adjacency list."""
    adj = {i: [] for i in range(3 * L * L)}
    for r in range(L):
        for c in range(L):
            for dr, dc, s_from, s_to in BOND_TABLE:
                i = flat_index(r, c, s_from, L)
                j = flat_index((r + dr) % L, (c + dc) % L, s_to, L)
                if j not in adj[i]:
                    adj[i].append(j)
    return adj


@lru_cache(maxsize=None)
def pair_displacements(L):
    """Minimum-image displacement (N, N, 2) and distance (N, N). Cached per L."""
    pos = site_positions(L).reshape(-1, 2)
    N = pos.shape[0]
    T1, T2 = L * A1, L * A2
    best = np.full((N, N, 2), np.inf)
    best_norm = np.full((N, N), np.inf)
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            d = pos[None, :, :] - pos[:, None, :] + m * T1 + n * T2
            nrm = np.linalg.norm(d, axis=-1)
            take = nrm < best_norm
            best_norm[take] = nrm[take]
            best[take] = d[take]
    return best, best_norm


@lru_cache(maxsize=None)
def geometry_tensors(L, device_str):
    rij, dist = pair_displacements(L)
    dev = torch.device(device_str)
    return (torch.tensor(rij, dtype=torch.float32, device=dev),
            torch.tensor(dist, dtype=torch.float32, device=dev))


# =============================================================================
# 2. DATA
# =============================================================================
H5_KEYS = ["U", "filling", "t1", "t2", "C", "n", "d"]


def load_samples(folder, n_sites, n_samples):
    """Load the first n_samples files for a given cluster size."""
    prefix = SIZE_PREFIX[n_sites]
    L_expected = int(round(np.sqrt(n_sites / 3)))
    samples = []
    for i in range(n_samples):
        path = os.path.join(folder, f"{prefix}{i:03d}.h5")
        with h5py.File(path, "r") as f:
            s = {k: f[k][()] for k in H5_KEYS}
            if "Lx" in f:
                L = int(f["Lx"][()])
            elif "L" in f:
                L = int(f["L"][()])
            else:
                L = L_expected
        assert 3 * L * L == n_sites, f"{path}: L={L} does not match {n_sites} sites"
        s["L"], s["path"] = L, path
        samples.append(s)
    print(f"  loaded {len(samples):3d} x {n_sites}-site samples from {folder}")
    return samples


def load_sources(sources):
    data = []
    for folder, n_sites, n_samples in sources:
        data += load_samples(folder, n_sites, n_samples)
    return data


def to_item(s):
    """Raw sample dict -> tensors on DEVICE."""
    L = s["L"]
    feats = torch.tensor([float(s["U"]), float(s["filling"]),
                          float(s["t1"]), float(s["t2"])])
    if NORMALIZE_INPUTS:
        feats = feats / torch.tensor(FEATURE_SCALE)
    # channels: [U, nu, t1, t2] repeated for the 3 sublattices -> 12 channels
    x = feats.repeat(3)[:, None, None].expand(12, L, L).contiguous()
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return dict(L=L, x=x.to(DEVICE), C=t(s["C"]).to(DEVICE),
                n=t(s["n"]).to(DEVICE), d=t(s["d"]).to(DEVICE))


def make_batches(items, batch_size, shuffle, rng):
    """Group items by lattice size, then stack into batches."""
    by_L = {}
    for it in items:
        by_L.setdefault(it["L"], []).append(it)
    batches = []
    for L, group in by_L.items():
        order = rng.permutation(len(group)) if shuffle else np.arange(len(group))
        rij, dist = geometry_tensors(L, str(DEVICE))
        for k in range(0, len(group), batch_size):
            chunk = [group[j] for j in order[k:k + batch_size]]
            B = len(chunk)
            batches.append(dict(
                L=L, B=B,
                x=torch.stack([c["x"] for c in chunk]),
                C=torch.stack([c["C"] for c in chunk]),
                n=torch.stack([c["n"] for c in chunk]),
                d=torch.stack([c["d"] for c in chunk]),
                rij=rij.unsqueeze(0).expand(B, -1, -1, -1),
                dist=dist.unsqueeze(0).expand(B, -1, -1),
            ))
    if shuffle:
        rng.shuffle(batches)
    return batches


# =============================================================================
# 3. MODEL
# =============================================================================
class KagomeCNN(nn.Module):
    """Fully convolutional encoder + pair decoder.

    Input : (B, f_in*3, L, L)
    Output: dict(n=(B,N), d=(B,N), C=(B,N,N)),  N = 3 L^2
    """

    def __init__(self, f_in=4, width=64, depth=4):
        super().__init__()
        layers, c_in = [], f_in * 3
        for _ in range(depth):
            layers += [nn.Conv2d(c_in, width * 3, kernel_size=3, padding=1,
                                 padding_mode="circular"), nn.SiLU()]
            c_in = width * 3
        self.body = nn.Sequential(*layers)
        self.width = width
        self.node_head = nn.Conv2d(width * 3, 2 * 3, kernel_size=1)
        self.pair = nn.Sequential(
            nn.Linear(2 * width + 3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, rij, dist):
        B, _, Lx, Ly = x.shape
        h = self.body(x)                                    # (B, 3W, L, L)

        # node observables; d_i <= n_i/2 enforced by construction
        node = self.node_head(h).view(B, 2, 3, Lx, Ly)
        n_i = torch.sigmoid(node[:, 0]) * 2.0
        d_i = torch.sigmoid(node[:, 1]) * (n_i / 2.0)

        # embeddings -> flat site list (index (r*L + c)*3 + s)
        hs = h.view(B, self.width, 3, Lx, Ly).permute(0, 3, 4, 2, 1)
        hs = hs.reshape(B, 3 * Lx * Ly, self.width)

        # pair decoder
        N = hs.shape[1]
        hi = hs[:, :, None, :].expand(B, N, N, self.width)
        hj = hs[:, None, :, :].expand(B, N, N, self.width)
        feat = torch.cat([hi, hj, rij, dist[..., None]], dim=-1)
        Cij = self.pair(feat).squeeze(-1)
        Cij = 0.5 * (Cij + Cij.transpose(1, 2))             # C_ij = C_ji

        # exact on-site identity <S_i.S_i> = 3/4 (n_i - 2 d_i)
        n_flat = n_i.permute(0, 2, 3, 1).reshape(B, N)
        d_flat = d_i.permute(0, 2, 3, 1).reshape(B, N)
        eye = torch.eye(N, device=x.device, dtype=Cij.dtype)
        Cij = Cij * (1.0 - eye) + torch.diag_embed(0.75 * (n_flat - 2 * d_flat))

        return dict(n=n_flat, d=d_flat, C=Cij)


class MultiTaskLoss(nn.Module):
    """Kendall, Gal & Cipolla (2018) learned task weighting.
    NOTE: the total can go negative/unbounded; judge convergence by the
    individual sub-losses, not by this total."""

    def __init__(self, n_tasks=3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for k, Lk in enumerate(losses):
            total = total + torch.exp(-self.log_sigma[k]) * Lk + self.log_sigma[k]
        return total


# =============================================================================
# 4. BASELINES  (computed per lattice size)
# =============================================================================
def _mse(pred_fn, data):
    return float(np.mean([np.mean((s["C"] - pred_fn(s)) ** 2) for s in data]))


def baseline_global_mean(train, test):
    mu = np.mean([s["C"].mean() for s in train])
    return _mse(lambda s: mu, test)


def baseline_mean_matrix(train, test):
    Cbar = np.mean([s["C"] for s in train], axis=0)
    return _mse(lambda s: Cbar, test)


def baseline_distance_U(train, test, L, nd=20, nu=12):
    """Predict C_ij from |r_ij| and U only (no angular/sublattice structure)."""
    _, dist = pair_displacements(L)
    dbins = np.linspace(0, dist.max() + 1e-9, nd + 1)
    Us = np.array([s["U"] for s in train], dtype=float)
    ubins = np.linspace(Us.min(), Us.max() + 1e-9, nu + 1)
    didx = np.clip(np.digitize(dist.ravel(), dbins) - 1, 0, nd - 1)

    table, count = np.zeros((nu, nd)), np.zeros((nu, nd))
    for s in train:
        u = np.clip(np.digitize(s["U"], ubins) - 1, 0, nu - 1)
        table[u] += np.bincount(didx, weights=s["C"].ravel(), minlength=nd)
        count[u] += np.bincount(didx, minlength=nd)
    table /= np.maximum(count, 1.0)

    def pred(s):
        u = np.clip(np.digitize(s["U"], ubins) - 1, 0, nu - 1)
        return table[u][didx].reshape(dist.shape)

    return _mse(pred, test)


def run_baselines(train, test):
    """Returns {L: L2 baseline MSE}."""
    l2 = {}
    for L in sorted({s["L"] for s in test}):
        tr = [s for s in train if s["L"] == L]
        te = [s for s in test if s["L"] == L]
        if not tr:
            print(f"  L={L}: no training samples of this size, baselines skipped")
            continue
        b0 = baseline_global_mean(tr, te)
        b1 = baseline_mean_matrix(tr, te)
        b2 = baseline_distance_U(tr, te, L)
        l2[L] = b2
        print(f"  L={L} ({3*L*L} sites, {len(tr)} train / {len(te)} test)")
        print(f"     L0 global mean        : {b0:.6f}")
        print(f"     L1 mean matrix (no U) : {b1:.6f}")
        print(f"     L2 distance + U       : {b2:.6f}   <-- CNN MUST BEAT THIS")
    return l2


# =============================================================================
# 5. TRAINING / EVALUATION
# =============================================================================
def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def train_step(model, mtl, optimizer, params, batch):
    sync()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    out = model(batch["x"], batch["rij"], batch["dist"])
    lC = F.mse_loss(out["C"], batch["C"])
    ln = F.mse_loss(out["n"], batch["n"])
    ld = F.mse_loss(out["d"], batch["d"])
    loss = mtl([lC, ld, ln])

    loss.backward()
    # returns the total gradient norm; only clips if GRAD_CLIP is set
    gnorm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP or float("inf"))
    optimizer.step()
    sync()
    dt = time.perf_counter() - t0

    L = batch["L"]
    return dict(L=L, N=3 * L * L, B=batch["B"], dt=dt, loss=loss.item(),
                lC=lC.item(), ln=ln.item(), ld=ld.item(), gnorm=gnorm.item())


@torch.no_grad()
def evaluate(model, items):
    """Mean sub-losses per lattice size on a held-out set."""
    model.eval()
    acc = {}
    for b in make_batches(items, 1, False, None):
        out = model(b["x"], b["rij"], b["dist"])
        vals = (F.mse_loss(out["C"], b["C"]).item(),
                F.mse_loss(out["n"], b["n"]).item(),
                F.mse_loss(out["d"], b["d"]).item())
        acc.setdefault(b["L"], []).append(vals)
    model.train()
    return {L: np.mean(v, axis=0) for L, v in acc.items()}


def print_eval(tag, result, l2):
    for L, (lC, ln, ld) in sorted(result.items()):
        base = f"  (L2 baseline {l2[L]:.6f})" if L in l2 else ""
        print(f"  [{tag}] L={L}: C {lC:.6f}{base} | n {ln:.6f} | d {ld:.6f}")


# =============================================================================
# 6. SUMMARY + PHYSICS CHECKS
# =============================================================================
def summarize(history, l2, total_time):
    print("\n" + "=" * 72)
    print("TRAINING SUMMARY")
    print("=" * 72)
    print(f"  total optimizer steps : {len(history)}")
    print(f"  total wall time       : {total_time:.1f} s")

    sizes = sorted({r["N"] for r in history})
    timed = [r for r in history if r["step"] > WARMUP_STEPS]
    mean_t = {}

    print("\n  --- time per step (warm-up excluded) ---")
    for N in sizes:
        ts = [r["dt"] for r in timed if r["N"] == N]
        if ts:
            mean_t[N] = float(np.mean(ts))
            print(f"  N={N:3d}: {1e3*mean_t[N]:8.2f} ms/step  "
                  f"(std {1e3*np.std(ts):.2f}, {len(ts)} steps)")
    if len(mean_t) >= 2:
        Ns = np.array(sorted(mean_t))
        ts = np.array([mean_t[N] for N in Ns])
        b, a = np.polyfit(Ns ** 2, ts, 1)
        print(f"  fit  t_step = a + b*N^2 :  a = {1e3*a:.2f} ms,  b = {1e3*b:.3e} ms/site^2")
        if len(Ns) >= 3:
            p, _ = np.polyfit(np.log(Ns), np.log(ts), 1)
            print(f"  log-log exponent p      : {p:.2f}  (~0 = overhead bound, ~2 = pair-decoder bound)")

    print("\n  --- gradient optimization ---")
    for N in sizes:
        recs = [r for r in history if r["N"] == N]
        L = recs[0]["L"]
        lC = np.array([r["lC"] for r in recs])
        gn = np.array([r["gnorm"] for r in recs])
        print(f"  N={N:3d}: lC first {lC[0]:.6f} -> last {lC[-1]:.6f}"
              f"   |grad| first {gn[0]:.3e} -> last {gn[-1]:.3e}"
              f"   (mean {gn.mean():.3e})")

        w = min(CONV_FIT_STEPS, len(lC))
        if w >= 5:
            k, _ = np.polyfit(np.arange(w), np.log(np.maximum(lC[:w], 1e-12)), 1)
            half = np.log(2) / abs(k) if k != 0 else float("inf")
            print(f"         early decay rate k = {k:+.5f}/step "
                  f"(loss halves every ~{half:.0f} steps)" if k < 0 else
                  f"         early decay rate k = {k:+.5f}/step (not decreasing)")

        target = l2.get(L, FALLBACK_TARGET_MSE)
        hit = next((r["step"] for r in recs if r["lC"] < target), None)
        if hit is None:
            print(f"         never reached target MSE {target:.6f}")
        else:
            n_hit = sum(1 for r in recs if r["step"] <= hit)
            est = n_hit * mean_t.get(N, float("nan"))
            print(f"         reached target MSE {target:.6f} at global step {hit} "
                  f"(~{n_hit} steps of this size, ~{est:.1f} s)")


def physics_report(model, items, raw):
    """Predicted vs target physics checks on the held-out set."""
    print("\n" + "=" * 72)
    print("PHYSICS SANITY CHECKS (test set)")
    print("=" * 72)
    model.eval()
    with torch.no_grad():
        for b, s in zip(make_batches(items, 1, False, None), raw):
            L = b["L"]
            out = model(b["x"], b["rij"], b["dist"])
            Cp = out["C"][0].cpu().numpy()
            Ct = s["C"]
            _, dist = pair_displacements(L)
            nn_mask = (dist > 0.4) & (dist < 0.6)
            diag_err = np.abs(np.diag(Cp) - 0.75 * (out["n"][0] - 2 * out["d"][0]).cpu().numpy()).max()
            print(f"  L={L}, U={float(s['U']):.2f}")
            print(f"     NN <S.S>   pred {Cp[nn_mask].mean():+.4f}   target {Ct[nn_mask].mean():+.4f}")
            print(f"     sum rule   pred {Cp.sum():+.4f}   target {Ct.sum():+.4f}")
            print(f"     |C - C^T|max {np.abs(Cp - Cp.T).max():.2e}   on-site identity err {diag_err:.2e}")
    model.train()


def plot_results(history):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for N in sorted({r["N"] for r in history}):
        recs = [r for r in history if r["N"] == N]
        st = [r["step"] for r in recs]
        ax[0].plot(st, [r["lC"] for r in recs], ".", ms=3, label=f"N={N}")
        ax[1].plot(st, [r["gnorm"] for r in recs], ".", ms=3, label=f"N={N}")
    ax[0].set(title="Spin-correlation MSE (per step)", xlabel="step", ylabel="MSE", yscale="log")
    ax[1].set(title="Gradient norm", xlabel="step", ylabel="|grad|", yscale="log")
    timed = [r for r in history if r["step"] > WARMUP_STEPS]
    Ns = sorted({r["N"] for r in timed})
    ax[2].plot(Ns, [1e3 * np.mean([r["dt"] for r in timed if r["N"] == N]) for N in Ns], "o-")
    ax[2].set(title="Mean time per step", xlabel="sites N", ylabel="ms")
    for a in ax[:2]:
        a.legend()
        a.grid(alpha=0.3, which="both")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"  saved plots -> {PLOT_PATH}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    print(f"device: {DEVICE}")

    # ---- geometry check ----
    print("=" * 72)
    print("KAGOME GEOMETRY CHECK")
    print("=" * 72)
    for L in (2, 3):
        N = 3 * L * L
        adj = build_adjacency(L)
        tri = sum(1 for i in adj for j in adj[i] for k in adj[j] if k in adj[i] and i < j < k)
        print(f"  L={L} ({N} sites): degrees {sorted({len(v) for v in adj.values()})} "
              f"(expect [4]),  triangles {tri} (expect {2*N//3})")

    # ---- data ----
    print("\n" + "=" * 72)
    print("DATA")
    print("=" * 72)
    train_raw = load_sources(TRAIN_SOURCES)
    test_raw = load_sources(TEST_SOURCES)
    overlap = {s["path"] for s in train_raw} & {s["path"] for s in test_raw}
    if overlap:
        print(f"  [WARNING] {len(overlap)} test file(s) are also in the training set "
              f"-> test numbers measure fit, not generalization: {sorted(overlap)}")
    train_items = [to_item(s) for s in train_raw]
    test_items = [to_item(s) for s in test_raw]

    # ---- baselines ----
    print("\n" + "=" * 72)
    print("BASELINES")
    print("=" * 72)
    l2 = run_baselines(train_raw, test_raw)

    # ---- model ----
    model = KagomeCNN(f_in=4, width=WIDTH, depth=DEPTH).to(DEVICE)
    if LOAD_CHECKPOINT:
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
            print(f"\n  loaded checkpoint {CHECKPOINT_PATH}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n  checkpoint not loaded ({type(e).__name__}); starting from scratch")
    mtl = MultiTaskLoss(n_tasks=3).to(DEVICE)
    params = list(model.parameters()) + list(mtl.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    # ---- initial evaluation ----
    print("\n" + "=" * 72)
    print(f"TRAINING: {EPOCHS} epochs, {len(train_items)} samples, batch size {BATCH_SIZE}")
    print("=" * 72)
    print_eval("test, before training", evaluate(model, test_items), l2)

    # ---- training loop ----
    history, step = [], 0
    t_train = time.perf_counter()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.perf_counter()
        ep_recs = []
        for batch in make_batches(train_items, BATCH_SIZE, True, rng):
            rec = train_step(model, mtl, optimizer, params, batch)
            step += 1
            rec["step"] = step
            history.append(rec)
            ep_recs.append(rec)

        mean = lambda key: float(np.mean([r[key] for r in ep_recs]))
        scheduler.step(mean("lC"))      # plateau on C-loss, not the unbounded MTL total

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:4d}/{EPOCHS} | step {step:6d} | "
                  f"C {mean('lC'):.6f} n {mean('ln'):.6f} d {mean('ld'):.6f} | "
                  f"total {mean('loss'):+.4f} | |grad| {mean('gnorm'):.3e} | "
                  f"lr {optimizer.param_groups[0]['lr']:.1e} | "
                  f"{time.perf_counter() - t_ep:.2f}s/epoch")
        if epoch % EVAL_EVERY == 0:
            print_eval(f"test, epoch {epoch}", evaluate(model, test_items), l2)
    total_time = time.perf_counter() - t_train

    # ---- reports ----
    summarize(history, l2, total_time)
    print("\n  final test performance:")
    print_eval("test, final", evaluate(model, test_items), l2)
    physics_report(model, test_items, test_raw)

    print("\n" + "=" * 72)
    print("OUTPUT")
    print("=" * 72)
    plot_results(history)
    if SAVE_PATH:
        os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  saved model -> {SAVE_PATH}")


if __name__ == "__main__":
    main()