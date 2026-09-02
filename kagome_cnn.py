"""
================================================================================
 kagome_cnn.py  --  size-agnostic, multi-task CNN for the kagome Hubbard model
================================================================================

WHY A CNN AND NOT A GNN
-----------------------
Kagome is NOT an arbitrary graph. It is a triangular Bravais lattice with a
3-site basis (sublattices A, B, C), so it maps exactly onto a tensor

        (L, L, 3)      L x L unit cells,  3 sublattice channels

Every kagome bond fits inside a 3x3 kernel window (verified: cell offsets are
only -1, 0, +1 in each direction).

A message-passing GNN aggregates over a site's neighbours in a
permutation-invariant way, which DISCARDS which neighbour is which. On kagome
the up-triangle and down-triangle bonds are geometrically distinct but identical
as graph edges, so the GNN sees {same,same,same,same} and the frustration
structure is destroyed at the first layer. (Formally: kagome and the square
lattice are both 4-regular, hence indistinguishable by 1-WL with uniform node
features; and message passing provably cannot count triangles.)

A convolution kernel does not throw that away. Kernel position (-1,0) is a
DIFFERENT learnable weight from (+1,0), so both triangles are visible for free.

SIZE-AGNOSTICISM
----------------
Comes free IF the network is fully convolutional. No dense layers anywhere.
Use mean pooling (not sum) for global quantities. Then the same weights run on
L=2, 4, 8 ... unit cells.

WHAT THIS FILE CONTAINS
-----------------------
  1. Kagome lattice geometry + bond table            (pure numpy, testable)
  2. Synthetic data generator (physics-flavoured)    -- so you can run TODAY,
                                                        before ED data exists
  3. The model                                       (PyTorch)
  4. Multi-task loss with hard-wired physics
  5. Baselines you MUST beat before believing anything
  6. Sanity checks against known kagome values

Requires: numpy, torch.   Run:  python kagome_cnn.py
================================================================================
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("[warn] torch not found -- geometry and baseline sections will still run")


# =============================================================================
# 1. KAGOME GEOMETRY
# =============================================================================

A1 = np.array([1.0, 0.0])                    # Bravais vectors (triangular)
A2 = np.array([0.5, np.sqrt(3) / 2])
BASIS = {0: np.array([0.0, 0.0]),            # A
         1: A1 / 2,                          # B
         2: A2 / 2}                          # C

# The 12 directed bonds, as (d_row, d_col, sublattice_from, sublattice_to).
# Verified by distance search: these are ALL nearest-neighbour bonds, degree 4.
BOND_TABLE = [
    (-1,  0, 0, 2), (-1, +1, 1, 2),
    ( 0, -1, 0, 1), ( 0,  0, 0, 1), ( 0,  0, 0, 2),
    ( 0,  0, 1, 0), ( 0,  0, 1, 2), ( 0,  0, 2, 0), ( 0,  0, 2, 1),
    ( 0, +1, 1, 0), (+1, -1, 2, 1), (+1,  0, 2, 0),
]


def site_positions(L):
    """Cartesian position of every site. Returns array of shape (L, L, 3, 2)."""
    pos = np.zeros((L, L, 3, 2))
    for r in range(L):
        for c in range(L):
            for s in range(3):
                pos[r, c, s] = r * A2 + c * A1 + BASIS[s]
    return pos


def flat_index(r, c, s, L):
    """Map (row, col, sublattice) -> flat site index in [0, 3L^2)."""
    return (r * L + c) * 3 + s


def build_adjacency(L):
    """Nearest-neighbour adjacency list, periodic. Use this to build ED
    Hamiltonians so the NN and the ED data agree on the lattice."""
    adj = {i: [] for i in range(3 * L * L)}
    for r in range(L):
        for c in range(L):
            for (dr, dc, s_from, s_to) in BOND_TABLE:
                i = flat_index(r, c, s_from, L)
                j = flat_index((r + dr) % L, (c + dc) % L, s_to, L)
                if j not in adj[i]:
                    adj[i].append(j)
    return adj


def pair_displacements(L):
    """Minimum-image relative displacement r_ij for every ordered pair.
    Returns (N, N, 2) with N = 3L^2. Used as input to the pair decoder."""
    pos = site_positions(L).reshape(-1, 2)          # (N, 2)
    N = pos.shape[0]
    # lattice vectors of the supercell
    T1, T2 = L * A1, L * A2
    best = np.full((N, N, 2), np.inf)
    best_norm = np.full((N, N), np.inf)
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            shift = m * T1 + n * T2
            d = pos[None, :, :] - pos[:, None, :] + shift
            nrm = np.linalg.norm(d, axis=-1)
            take = nrm < best_norm
            best_norm[take] = nrm[take]
            best[take] = d[take]
    return best, best_norm


# =============================================================================
# 2. SYNTHETIC DATA  (placeholder -- replace with ED as soon as it exists)
# =============================================================================
# Physics put in deliberately, so you can verify the whole pipeline works
# BEFORE spending compute on exact diagonalization:
#   - correlations decay exponentially, xi ~ 1 lattice spacing
#   - nearest neighbour NEGATIVE  (target ~ -0.22 in the Heisenberg limit)
#   - second neighbour small POSITIVE
#   - double occupancy falls as U/t grows
# If the model cannot learn THIS, it will not learn ED data either.

def synthetic_sample(L, U, filling, rng):
    rij, dist = pair_displacements(L)
    N = dist.shape[0]

    # --- local observables ---
    n_i = np.full(N, filling)
    d_i = np.full(N, 0.25 * filling ** 2 / (1.0 + 0.35 * U))   # double occupancy

    # --- spin correlations ---
    xi = 1.0 + 2.0 / (1.0 + 0.3 * U)          # correlation length shrinks with U
    with np.errstate(divide="ignore"):
        env = np.exp(-dist / xi)

    # radial sign structure: NN negative, 2nd neighbour positive, alternating
    radial = np.cos(2 * np.pi * (dist - 0.5))

    # ANGULAR structure with the p6m symmetry of kagome.  This is what splits
    # pairs at the SAME distance into inequivalent sets (as really happens for
    # third neighbours on kagome: four antiferromagnetic + two ferromagnetic).
    # It is also what makes the distance-only baseline beatable -- without it
    # the benchmark is vacuous.
    theta = np.arctan2(rij[..., 1], rij[..., 0])
    angular = 0.35 * np.cos(6 * theta)

    C = -env * (radial + angular)
    C = 0.5 * (C + C.T)                        # correlations are symmetric

    # calibrate so the nearest-neighbour value approaches the known kagome
    # Heisenberg result  <S.S>_nn ~ -0.22  as U grows
    nn_mask = (dist > 0.4) & (dist < 0.6)
    target_nn = -0.22 * U / (U + 2.0)
    C *= target_nn / C[nn_mask].mean()

    C += 0.002 * rng.standard_normal(C.shape)  # measurement-like noise
    C = 0.5 * (C + C.T)

    # exact on-site identity  <S_i.S_i> = (3/4)(n_i - 2 d_i)   -- set LAST
    np.fill_diagonal(C, 0.75 * (n_i - 2 * d_i))
    return dict(U=U, filling=filling, L=L, n=n_i, d=d_i, C=C)


def make_dataset(L, n_samples, rng):
    out = []
    for _ in range(n_samples):
        U = rng.uniform(0.5, 12.0)
        nu = rng.uniform(0.7, 1.0)
        out.append(synthetic_sample(L, U, nu, rng))
    return out


# =============================================================================
# 3. THE MODEL
# =============================================================================

if HAVE_TORCH:

    class KagomeCNN(nn.Module):
        """Fully convolutional. NO dense layers -> works at any L.

        Input : per-site scalars packed as (B, F_in*3, L, L)
        Output: node embeddings h of shape (B, C, L, L, 3)
        """

        def __init__(self, f_in=2, width=64, depth=4):
            super().__init__()
            layers = []
            c_in = f_in * 3                       # 3 sublattices folded into channels
            for _ in range(depth):
                layers += [
                    nn.Conv2d(c_in, width * 3, kernel_size=3,
                              padding=1, padding_mode="circular"),  # periodic BC
                    nn.SiLU(),
                ]
                c_in = width * 3
            self.body = nn.Sequential(*layers)
            self.width = width

            # ---- node head: 1x1 conv, so still fully convolutional ----
            # predicts  [n_i, d_i]  only.  <S_i.S_i> is DERIVED, never predicted.
            self.node_head = nn.Conv2d(width * 3, 2 * 3, kernel_size=1)

            # ---- pair decoder: f(h_i, h_j, r_ij) -> C_ij ----
            # This replaces an N x N output layer, which would pin the size.
            self.pair = nn.Sequential(
                nn.Linear(2 * width + 3, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x, rij, dist):
            """
            x    : (B, F_in*3, L, L)
            rij  : (B, N, N, 2) minimum-image displacements
            dist : (B, N, N)    distances
            """
            B, _, L, _ = x.shape
            h = self.body(x)                              # (B, 3W, L, L)

            # ---- node observables ----
            node = self.node_head(h)                      # (B, 6, L, L)
            node = node.view(B, 2, 3, L, L)
            n_i = torch.sigmoid(node[:, 0]) * 2.0         # density in [0, 2]
            d_i = torch.sigmoid(node[:, 1])               # double occ in [0, 1]
            # physical constraint: d_i <= n_i/2
            d_i = torch.minimum(d_i, n_i / 2)

            # ---- reshape embeddings to a flat site list ----
            hs = h.view(B, self.width, 3, L, L)           # (B, W, 3, L, L)
            hs = hs.permute(0, 3, 4, 2, 1).reshape(B, 3 * L * L, self.width)

            # ---- pair decoder over all pairs ----
            N = hs.shape[1]
            hi = hs[:, :, None, :].expand(B, N, N, self.width)
            hj = hs[:, None, :, :].expand(B, N, N, self.width)
            feat = torch.cat([hi, hj, rij, dist[..., None]], dim=-1)
            Cij = self.pair(feat).squeeze(-1)             # (B, N, N)
            Cij = 0.5 * (Cij + Cij.transpose(1, 2))       # symmetry: C_ij = C_ji

            # ---- HARD-WIRED PHYSICS ----
            n_flat = n_i.permute(0, 2, 3, 1).reshape(B, N)
            d_flat = d_i.permute(0, 2, 3, 1).reshape(B, N)
            diag = 0.75 * (n_flat - 2 * d_flat)           # exact on-site identity
            Cij = Cij - torch.diag_embed(torch.diagonal(Cij, dim1=1, dim2=2))
            Cij = Cij + torch.diag_embed(diag)

            return dict(n=n_flat, d=d_flat, C=Cij)


    # =========================================================================
    # 4. LOSS  --  multi-task with learned uncertainty weighting
    # =========================================================================

    class MultiTaskLoss(nn.Module):
        """Kendall, Gal & Cipolla (CVPR 2018): learn a per-task noise term
        instead of hand-tuning weights. Prevents the largest-scale target from
        swamping the small correlators, which is the usual failure mode."""

        def __init__(self, n_tasks=3):
            super().__init__()
            self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

        def forward(self, losses):
            total = 0.0
            for k, Lk in enumerate(losses):
                total = total + torch.exp(-self.log_sigma[k]) * Lk + self.log_sigma[k]
            return total


    def correlation_loss(pred, target, eps=0.02):
        """Correlations span orders of magnitude and change sign, so plain MSE
        only ever learns the nearest-neighbour term. Weight by inverse magnitude
        so a 10% error on a small distant correlator counts like a 10% error on
        the dominant one."""
        w = 1.0 / (target.abs() + eps)
        return (w * (pred - target) ** 2).mean()


    def energy_from_observables(d, U, kinetic):
        """E = -t sum <c^dag c>  +  U sum <n_up n_dn>.
        Compute the energy, never predict it. Removes a head, removes a
        competing gradient, and gives an exact constraint for free."""
        return kinetic + U * d.sum(dim=1)


# =============================================================================
# 5. BASELINES  --  run these FIRST. If the CNN does not beat them, stop.
# =============================================================================

def _mse(pred_fn, data):
    return float(np.mean([np.mean((s["C"] - pred_fn(s)) ** 2) for s in data]))


def baseline_global_mean(train, test):
    """Level 0: one number for every entry. The floor."""
    mu = np.mean([s["C"].mean() for s in train])
    return _mse(lambda s: mu, test)


def baseline_mean_matrix(train, test):
    """Level 1: the sample-averaged matrix. Captures spatial structure but is
    blind to U."""
    Cbar = np.mean([s["C"] for s in train], axis=0)
    return _mse(lambda s: Cbar, test)


def baseline_distance_U(train, test, L, nd=20, nu=12):
    """Level 2: THE BASELINE THAT MATTERS.  Predicts C_ij from |r_i - r_j| AND
    U, i.e. everything except the angular / sublattice structure.  A model that
    cannot beat this has learned nothing that the radial profile does not
    already give away -- which is exactly the failure mode of a plain GNN."""
    _, dist = pair_displacements(L)
    dbins = np.linspace(0, dist.max() + 1e-9, nd + 1)
    Us = np.array([s["U"] for s in train])
    ubins = np.linspace(Us.min(), Us.max() + 1e-9, nu + 1)
    didx = np.clip(np.digitize(dist.ravel(), dbins) - 1, 0, nd - 1)

    table = np.zeros((nu, nd)); count = np.zeros((nu, nd))
    for s in train:
        u = np.clip(np.digitize(s["U"], ubins) - 1, 0, nu - 1)
        vals = s["C"].ravel()
        np.add.at(table[u], didx, vals)
        np.add.at(count[u], didx, 1.0)
    table /= np.maximum(count, 1.0)

    def pred(s):
        u = np.clip(np.digitize(s["U"], ubins) - 1, 0, nu - 1)
        return table[u][didx].reshape(dist.shape)
    return _mse(pred, test)


# =============================================================================
# 6. SANITY CHECKS  --  physics, not loss curves
# =============================================================================

def sanity_checks(sample, L, verbose=True):
    """Run these on EVERY model output. A good loss curve means nothing if
    these fail."""
    C = sample["C"]
    _, dist = pair_displacements(L)
    nn_mask = (dist > 0.4) & (dist < 0.6)          # nearest neighbours
    results = {
        "nn_correlation": C[nn_mask].mean(),        # target ~ -0.22 (Heisenberg)
        "sum_rule": C.sum(),                        # = 0 for a singlet
        "symmetry_violation": np.abs(C - C.T).max(),# must be ~0
        "diag_identity_error": np.abs(
            np.diag(C) - 0.75 * (sample["n"] - 2 * sample["d"])).max(),
    }
    if verbose:
        print("  nearest-neighbour <S.S>   : "
              f"{results['nn_correlation']:+.4f}   (Heisenberg limit ~ -0.22)")
        print(f"  sum rule  sum_ij <S_i.S_j>: {results['sum_rule']:+.4f}   "
              "(0 for a singlet; NOT satisfied by the")
        print("                              synthetic data above -- this check is"
              " for real ED data)")
        print(f"  |C - C^T|_max             : {results['symmetry_violation']:.2e}")
        print(f"  on-site identity error    : {results['diag_identity_error']:.2e}")
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L = 2                                   # 2x2 unit cells = 12 sites (ED-friendly)
    N = 3 * L * L

    print("=" * 72)
    print(f"KAGOME GEOMETRY CHECK   L={L}  ->  {N} sites")
    print("=" * 72)
    adj = build_adjacency(L)
    degs = sorted({len(v) for v in adj.values()})
    print(f"  degrees present            : {degs}      (kagome must be 4)")
    tri = sum(1 for i in adj for j in adj[i] for k in adj[j]
              if k in adj[i] and i < j < k)
    print(f"  triangles                  : {tri}      (expect 2N/3 = {2*N//3})")

    print()
    print("=" * 72)
    print("BASELINES  --  the CNN must beat the distance baseline")
    print("=" * 72)
    train = make_dataset(L, 400, rng)
    test = make_dataset(L, 100, rng)
    b0 = baseline_global_mean(train, test)
    b1 = baseline_mean_matrix(train, test)
    b2 = baseline_distance_U(train, test, L)
    print(f"  L0 global mean             : {b0:.6f}")
    print(f"  L1 mean matrix (no U)      : {b1:.6f}")
    print(f"  L2 distance + U            : {b2:.6f}   <-- BEAT THIS")
    print(f"     L2 explains {100*(1-b2/b0):.1f}% of the variance;")
    print(f"     the remainder is the angular / sublattice structure,")
    print(f"     which is precisely what a plain GNN cannot represent.")

    print()
    print("=" * 72)
    print("SANITY CHECKS on a synthetic sample")
    print("=" * 72)
    sanity_checks(train[0], L)

    if HAVE_TORCH:
        print()
        print("=" * 72)
        print("MODEL: size-agnosticism test (same weights, different L)")
        print("=" * 72)
        model = KagomeCNN()
        for Ltest in (2, 3, 4):
            Nt = 3 * Ltest * Ltest
            x = torch.randn(1, 2 * 3, Ltest, Ltest)
            rij, dist = pair_displacements(Ltest)
            out = model(x,
                        torch.tensor(rij, dtype=torch.float32)[None],
                        torch.tensor(dist, dtype=torch.float32)[None])
            print(f"  L={Ltest}  sites={Nt:3d}  ->  C shape {tuple(out['C'].shape)}  OK")
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  parameter count is independent of L: {n_par:,}")
