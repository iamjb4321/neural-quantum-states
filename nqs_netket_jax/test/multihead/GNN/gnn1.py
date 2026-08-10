import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tenpy.tools.hdf5_io as h5_tenpy
from tenpy.algorithms.exact_diag import get_full_wavefunction

# ==========================================
# 1. HDF5 DATASET LOADER (Restored for actual H5 files)
# ==========================================
'''
try:
    
    HAS_TENPY = True
except ImportError:
    HAS_TENPY = False
    '''

def load_kagome_dataset(h5_filepath):
    print(f"--- Loading Kagome Dataset from '{h5_filepath}' ---")

    if not os.path.exists(h5_filepath):
        raise FileNotFoundError(f"Could not locate file: {h5_filepath}")

    with h5py.File(h5_filepath, "r") as f:
        # --- A. Read Model Parameters ---
        p = f["model_params"]
        Lx = int(p["Lx"][()]) if "Lx" in p else 1
        Ly = int(p["Ly"][()]) if "Ly" in p else 1

        U = float(p["U"][()]) if "U" in p else 0.0
        t1 = float(p["t1"][()]) if "t1" in p else 1.0
        t2 = float(p["t2"][()]) if "t2" in p else 0.0
        theta_x = float(p["theta_x"][()]) if "theta_x" in p else 0.0
        theta_y = float(p["theta_y"][()]) if "theta_y" in p else 0.0

        param_vector = np.array([U, t1, t2, theta_x, theta_y], dtype=np.float32)

        # --- B. Read Observables ---
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

        # --- C. Read or Deserialise Wavefunction ---
        y_psi = None
        if "psi" in f:
            psi_mps = h5_tenpy.load_from_hdf5(f, "psi")
            psi_dense = get_full_wavefunction(psi_mps)#.to_ndarray()
            y_psi = psi_dense.reshape(-1).astype(np.float32)
        elif "y_psi" in f:
            y_psi = np.array(f["y_psi"], dtype=np.float32)
        else:
            print("y_psi/psi not found")

    # --- D. Build Basis Matrix X ---
    if y_psi is not None:
        dim = len(y_psi)
        y_psi = y_psi / np.linalg.norm(y_psi)

        basis_indices = np.arange(dim)
        binary_matrix = (basis_indices[:, None] >> np.arange(num_sites)[::-1]) & 1
        X_data = (2 * binary_matrix - 1).astype(np.float32)
    else:
        # Fallback if no psi is found (just creating dummy inputs matching params)
        X_data = np.repeat(param_vector[None, :], 16, axis=0) 

    print(f"Loaded successfully! Lattice Sites: {num_sites}")
    return X_data, y_psi, observables, param_vector, num_sites

# Helper: Build Kagome Adjacency Matrix 
def build_adjacency_matrix(num_sites):
    """Creates an adjacency matrix for the graph."""
    adj = np.zeros((num_sites, num_sites), dtype=np.float32)
    
    # Simple nearest-neighbor fallback for testing. 
    # For a 3-site system, this naturally forms a single triangle.
    for i in range(num_sites):
        adj[i, (i+1)%num_sites] = 1.0
        adj[(i+1)%num_sites, i] = 1.0
        
    return jnp.array(adj)

# ==========================================
# 2. GNN MODULES (SIZE-AGNOSTIC)
# ==========================================
class GCNLayer(nn.Module):
    """A standard Graph Convolutional Layer."""
    features: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        # node_feats: [Batch, N, in_features]
        # adj_matrix: [Batch, N, N]
        
        # 1. Apply Dense transformation to node features
        x = nn.Dense(self.features)(node_feats) 
        
        # 2. Message Passing: Aggregate from neighbors using adjacency
        # jnp.matmul naturally handles batched matrix multiplication
        x = jnp.matmul(adj_matrix, x) 
        return nn.tanh(x)


class NodePropertyHead(nn.Module):
    """Applies a shared MLP to each node independently for 1D vectors (e.g., density)."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_feats):
        # node_feats: [Batch, N, Features]
        x = nn.Dense(self.hidden_dim)(node_feats)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)  # [Batch, N, 1]
        return jnp.squeeze(x, axis=-1)  # [Batch, N]


class BilinearCorrelationHead(nn.Module):
    """Computes pairwise interactions resulting in an N x N symmetric matrix."""
    @nn.compact
    def __call__(self, node_feats):
        # node_feats: [Batch, N, Features]
        feat_dim = node_feats.shape[-1]
        
        # Learnable interaction matrix W (Features x Features)
        # Notice this does NOT depend on N!
        W = self.param('W', nn.initializers.normal(stddev=0.1), (feat_dim, feat_dim))
        
        # Compute H * W * H^T for all batches
        # out shape: [Batch, N, N]
        raw_corr = jnp.einsum('bni,ij,bmj->bnm', node_feats, W, node_feats)
        
        # Enforce exact matrix symmetry C_ij = C_ji
        return 0.5 * (raw_corr + jnp.swapaxes(raw_corr, 1, 2))


class KagomeGNN_NQS(nn.Module):
    """The unified size-agnostic GNN architecture."""
    hidden_dim: int = 128
    head_dim: int = 64

    @nn.compact
    def __call__(self, x_input, adj_matrix):
        # Format input to Node Features: [Batch, N] -> [Batch, N, 1]
        node_feats = jnp.expand_dims(x_input, axis=-1)
        
        # Shared Feature Extractor Trunk (Message Passing)
        h = GCNLayer(self.hidden_dim)(node_feats, adj_matrix)
        h = GCNLayer(self.hidden_dim)(h, adj_matrix)

        # Head 0: Wavefunction Amplitudes (Global Pooling)
        h_global = jnp.mean(h, axis=1) # [Batch, hidden_dim]
        log_psi = jnp.squeeze(nn.Dense(1)(h_global), axis=-1)

        # Multi-Layer Heads routed dynamically
        pred_heads = {
            "charge_density": NodePropertyHead(hidden_dim=self.head_dim)(h),
            "spin_corr": BilinearCorrelationHead()(h)
        }

        return log_psi, pred_heads


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_kagome_nqs(
    X_data, A_data, y_psi, observables_true, params, opt_state, optimizer, model, num_epochs=1500, w_psi=1.0
):
    print("--- Training Size-Agnostic GNN NQS ---")

    # Add batch dimension if missing for observables
    targets_jax = {k: jnp.array(v)[None, ...] for k, v in observables_true.items()}
    X_inputs = jnp.array(X_data)
    A_inputs = jnp.repeat(A_data[None, ...], X_inputs.shape[0], axis=0) # [Batch, N, N]
    y_targets = jnp.array(y_psi)

    @jax.jit
    def loss_fn(params, x_batch, adj_batch, y_batch):
        log_psi, pred_heads = model.apply({"params": params}, x_batch, adj_batch)

        # Wavefunction Loss
        log_psi_shifted = log_psi - jnp.max(log_psi)
        psi = jnp.exp(log_psi_shifted)
        overlap = jnp.dot(psi, y_batch)
        norm_model = jnp.linalg.norm(psi)
        norm_true = jnp.linalg.norm(y_batch)
        loss_psi = 1.0 - ((overlap**2) / (norm_model**2 * norm_true**2 + 1e-12))

        # Observables Loss
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

    # Training iterations
    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state, X_inputs, A_inputs, y_targets)
        loss_psi, head_losses, pred_heads = aux

        if (epoch + 1) % 150 == 0:
            loss_str = " | ".join([f"{k[:6]}: {v:.2e}" for k, v in head_losses.items()])
            print(f"Epoch {epoch+1:4d} | Total Loss: {total_loss:.6f} | {loss_str}")
            print(f"Predict heads: {pred_heads}")

    # Remove batch dim for final display output
    pred_heads_squeezed = {k: np.array(v[0]) for k, v in pred_heads.items()}
    return params, opt_state, pred_heads_squeezed, head_losses


# ==========================================
# 4. ENTRY POINT (Processing Multiple Sizes)
# ==========================================
if __name__ == "__main__":
    
    # Initialize the model completely independent of 'num_sites'
    model = KagomeGNN_NQS(hidden_dim=128, head_dim=64)
    rng = jax.random.PRNGKey(42)
    
    # We can initialize weights using a dummy small graph (e.g., 3 sites)
    dummy_x = jnp.ones((1, 3))
    dummy_a = jnp.ones((1, 3, 3))
    params = model.init(rng, dummy_x, dummy_a)["params"]
    
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=3000, alpha=0.01)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # List of files representing different lattice sizes
    # We can use the SAME 'params' object to train across all of them!

    for i in range(5):
        h5_file = r"C:\Users\jj9ba\Research\DREU QIS-AI\AI Model\neural-quantum-states\kagome_data\three_site_kagome_"+f"{i:03d}.h5"
        print(f"\n{'='*65}\nProcessing File: {h5_file}\n{'='*65}")
        
        # Load dataset
        X_data, y_psi, observables_true, param_vector, num_sites = load_kagome_dataset(h5_file)
        
        # Build Adjacency Matrix for this specific graph size
        A_data = build_adjacency_matrix(num_sites)

        # Train the model. It updates the exact same 'params' dictionary.
        # JAX will automatically re-compile 'train_step' under the hood if 'num_sites' changes, 
        # but the actual Neural Network weights are preserved and transferred seamlessly.
        params, opt_state, pred_heads, head_losses = train_kagome_nqs(
            X_data, A_data, y_psi, observables_true, params, opt_state, optimizer, model, num_epochs=300
        )