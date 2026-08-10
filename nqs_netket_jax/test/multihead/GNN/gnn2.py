import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# Try importing TeNPy helper for native MPS deserialization
try:
    import tenpy.tools.hdf5_io as h5_tenpy
    from tenpy.algorithms.exact_diag import get_full_wavefunction
    HAS_TENPY = True
except ImportError:
    HAS_TENPY = False


# ==========================================
# 1. KAGOME ADJACENCY MATRIX GENERATOR
# ==========================================
def build_kagome_adjacency(Lx: int, Ly: int, periodic: bool = True) -> jnp.ndarray:
    """
    Builds the exact 2D Kagome lattice adjacency matrix.
    N = 3 * Lx * Ly total lattice sites across 3-site unit cells (A, B, C).
    """
    num_unit_cells = Lx * Ly
    num_sites = 3 * num_unit_cells
    adj = np.zeros((num_sites, num_sites), dtype=np.float32)

    def get_site_index(x, y, sublattice):
        if periodic:
            x = x % Lx
            y = y % Ly
        else:
            if x < 0 or x >= Lx or y < 0 or y >= Ly:
                return -1
        uc_index = x + y * Lx
        return 3 * uc_index + sublattice

    def add_edge(site1, site2):
        if site1 != -1 and site2 != -1:
            adj[site1, site2] = 1.0
            adj[site2, site1] = 1.0

    for y in range(Ly):
        for x in range(Lx):
            A = get_site_index(x, y, 0)
            B = get_site_index(x, y, 1)
            C = get_site_index(x, y, 2)

            # Intra-unit cell bonds (internal triangle)
            add_edge(A, B)
            add_edge(B, C)
            add_edge(C, A)

            # Inter-unit cell bonds
            B_left = get_site_index(x - 1, y, 1)
            add_edge(A, B_left)

            A_down = get_site_index(x, y - 1, 0)
            add_edge(C, A_down)

            C_diag = get_site_index(x - 1, y - 1, 2)
            add_edge(B, C_diag)

    return jnp.array(adj)


# ==========================================
# 2. HDF5 DATASET LOADER
# ==========================================
def load_kagome_dataset(h5_filepath):
    print(f"--- Loading Kagome Dataset from '{h5_filepath}' ---")

    if not os.path.exists(h5_filepath):
        raise FileNotFoundError(f"Could not locate file: {h5_filepath}")

    with h5py.File(h5_filepath, "r") as f:
        p = f["model_params"]
        Lx = int(p["Lx"][()]) if "Lx" in p else 1
        Ly = int(p["Ly"][()]) if "Ly" in p else 1

        U = float(p["U"][()]) if "U" in p else 0.0
        t1 = float(p["t1"][()]) if "t1" in p else 1.0
        t2 = float(p["t2"][()]) if "t2" in p else 0.0
        theta_x = float(p["theta_x"][()]) if "theta_x" in p else 0.0
        theta_y = float(p["theta_y"][()]) if "theta_y" in p else 0.0

        param_vector = np.array([U, t1, t2, theta_x, theta_y], dtype=np.float32)

        obs = f["observables"]

        def get_obs(key):
            if key in obs:
                return np.array(obs[key], dtype=np.float32)
            raise KeyError(f"Required observable '{key}' not found in HDF5 file.")

        density_true = np.squeeze(get_obs("charge_density"))
        spin_corr_true = np.squeeze(get_obs("spin_corr"))

        num_sites = len(density_true)

        observables = {
            "charge_density": density_true,
            "spin_corr": spin_corr_true,
        }

        # Wavefunction Deserialization
        y_psi = None
        if "psi" in f and HAS_TENPY:
            psi_mps = h5_tenpy.load_from_hdf5(f, "psi")
            psi_dense = get_full_wavefunction(psi_mps)
            y_psi = np.array(psi_dense, dtype=np.float32).reshape(-1)
        elif "y_psi" in f:
            y_psi = np.array(f["y_psi"], dtype=np.float32)

    # Build spin basis configuration matrix
    if y_psi is not None:
        dim = len(y_psi)
        y_psi = y_psi / np.linalg.norm(y_psi)

        basis_indices = np.arange(dim)
        binary_matrix = (basis_indices[:, None] >> np.arange(num_sites)[::-1]) & 1
        X_data = (2 * binary_matrix - 1).astype(np.float32)
    else:
        X_data = np.repeat(param_vector[None, :], 16, axis=0)

    print(f"Loaded successfully! Lattice Sites: {num_sites} | (Lx={Lx}, Ly={Ly})")
    return X_data, y_psi, observables, param_vector, num_sites, Lx, Ly


# ==========================================
# 3. GNN MODULES (SIZE-AGNOSTIC)
# ==========================================
class GCNLayer(nn.Module):
    """Message passing layer aggregating node representations over the Kagome graph."""
    features: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        x = nn.Dense(self.features)(node_feats)
        x = jnp.matmul(adj_matrix, x)  # Neighbor aggregation
        return nn.tanh(x)


class NodePropertyHead(nn.Module):
    """Shared per-node MLP predicting 1D site vector properties (e.g. charge density)."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_feats):
        x = nn.Dense(self.hidden_dim)(node_feats)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class BilinearCorrelationHead(nn.Module):
    """Size-agnostic head generating symmetric N x N correlation matrices."""
    @nn.compact
    def __call__(self, node_feats):
        feat_dim = node_feats.shape[-1]
        W = self.param('W', nn.initializers.normal(stddev=0.1), (feat_dim, feat_dim))
        
        # Tensor contraction H * W * H^T
        raw_corr = jnp.einsum('bni,ij,bmj->bnm', node_feats, W, node_feats)
        
        # Enforce C_ij = C_ji
        return 0.5 * (raw_corr + jnp.swapaxes(raw_corr, 1, 2))


class KagomeGNN_NQS(nn.Module):
    """Unified size-agnostic GNN architecture."""
    hidden_dim: int = 128
    head_dim: int = 64

    @nn.compact
    def __call__(self, x_input, adj_matrix):
        node_feats = jnp.expand_dims(x_input, axis=-1)
        
        # Message passing trunk
        h = GCNLayer(self.hidden_dim)(node_feats, adj_matrix)
        h = GCNLayer(self.hidden_dim)(h, adj_matrix)

        # Head 0: Quantum State Amplitudes log|psi|
        h_global = jnp.mean(h, axis=1)
        log_psi = jnp.squeeze(nn.Dense(1)(h_global), axis=-1)

        # Observable Prediction Heads
        pred_heads = {
            "charge_density": NodePropertyHead(hidden_dim=self.head_dim)(h),
            "spin_corr": BilinearCorrelationHead()(h)
        }

        return log_psi, pred_heads


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_kagome_nqs(
    X_data, A_data, y_psi, observables_true, params, opt_state, optimizer, model, num_epochs=1500, w_psi=1.0
):
    print("--- Training Size-Agnostic GNN NQS ---")

    targets_jax = {k: jnp.array(v)[None, ...] for k, v in observables_true.items()}
    X_inputs = jnp.array(X_data)
    A_inputs = jnp.repeat(A_data[None, ...], X_inputs.shape[0], axis=0)
    y_targets = jnp.array(y_psi) if y_psi is not None else None

    @jax.jit
    def loss_fn(params, x_batch, adj_batch, y_batch):
        log_psi, pred_heads = model.apply({"params": params}, x_batch, adj_batch)

        # Overlap Loss
        if y_batch is not None:
            log_psi_shifted = log_psi - jnp.max(log_psi)
            psi = jnp.exp(log_psi_shifted)
            overlap = jnp.dot(psi, y_batch)
            norm_model = jnp.linalg.norm(psi)
            norm_true = jnp.linalg.norm(y_batch)
            loss_psi = 1.0 - ((overlap**2) / (norm_model**2 * norm_true**2 + 1e-12))
        else:
            loss_psi = 0.0

        # Physical Properties MSE Loss
        total_obs_loss = 0.0
        head_losses = {}
        for name, target in targets_jax.items():
            l_head = jnp.mean((pred_heads[name] - target) ** 2)
            head_losses[name] = l_head
            total_obs_loss += 0.5 * l_head

        total_loss = w_psi * loss_psi + total_obs_loss
        return total_loss, (loss_psi, head_losses, pred_heads)

    @jax.jit
    def train_step(params, opt_state, x_batch, adj_batch, y_batch):
        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_batch, adj_batch, y_batch
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, aux

    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state, X_inputs, A_inputs, y_targets)
        loss_psi, head_losses, pred_heads = aux

        if (epoch + 1) % 150 == 0:
            loss_str = " | ".join([f"{k[:6]}: {v:.2e}" for k, v in head_losses.items()])
            print(f"Epoch {epoch+1:4d} | Total Loss: {total_loss:.6f} | {loss_str}")

    pred_heads_squeezed = {k: np.array(v[0]) for k, v in pred_heads.items()}
    return params, opt_state, pred_heads_squeezed, head_losses


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    model = KagomeGNN_NQS(hidden_dim=128, head_dim=64)
    rng = jax.random.PRNGKey(42)

    # Initialize GNN weights using a standard 3-site dummy graph
    dummy_x = jnp.ones((1, 3))
    dummy_a = jnp.ones((1, 3, 3))
    params = model.init(rng, dummy_x, dummy_a)["params"]

    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=1500, alpha=0.01)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # Sequence of HDF5 files representing different state points or lattice sizes
    data_files = ["three_site_Kagome_Hubbard_296.h5"]

    for h5_file in data_files:
        if not os.path.exists(h5_file):
            print(f"Skipping missing file: {h5_file}")
            continue

        print(f"\n{'='*65}\nProcessing File: {h5_file}\n{'='*65}")

        # 1. Load Data
        X_data, y_psi, observables_true, param_vector, num_sites, Lx, Ly = load_kagome_dataset(h5_file)

        # 2. Build Exact 2D Kagome Adjacency Matrix
        A_data = build_kagome_adjacency(Lx=Lx, Ly=Ly, periodic=True)

        # 3. Train/Transfer Model Weights Across Datasets
        params, opt_state, pred_heads, head_losses = train_kagome_nqs(
            X_data, A_data, y_psi, observables_true, params, opt_state, optimizer, model, num_epochs=600
        )

        # 4. Display Results
        print("\n[Final Mean Squared Errors]:")
        for k, v in head_losses.items():
            print(f"  - Head '{k}': MSE = {float(v):.4e}")

        if "spin_corr" in pred_heads:
            print("\n[Predicted Symmetrized Spin Correlation Matrix (3x3 Block)]:")
            print(np.array2string(pred_heads["spin_corr"][:3, :3], precision=4, suppress_small=True))
            