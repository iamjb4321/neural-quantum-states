import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt

# ==========================================
# 1. KAGOME ADJACENCY MATRIX GENERATOR
# ==========================================
def build_kagome_adjacency(Lx: int, Ly: int, periodic: bool = True) -> jnp.ndarray:
    """Builds the exact 2D Kagome lattice adjacency matrix."""
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

            # Intra-unit cell
            add_edge(A, B)
            add_edge(B, C)
            add_edge(C, A)

            # Inter-unit cell
            B_left = get_site_index(x - 1, y, 1)
            add_edge(A, B_left)
            A_down = get_site_index(x, y - 1, 0)
            add_edge(C, A_down)
            C_diag = get_site_index(x - 1, y - 1, 2)
            add_edge(B, C_diag)

    return jnp.array(adj)


# ==========================================
# 2. HDF5 DATASET LOADER (SURROGATE VERSION)
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
        
        # Try to load energy; default to 0.0 if not present for testing
        if "energy" in f:
            energy_true = float(f["energy"][()])
        else:
            energy_true = 0.0

        num_sites = len(density_true)

        observables = {
            "charge_density": density_true,
            "spin_corr": spin_corr_true,
            "energy": np.array([energy_true], dtype=np.float32)
        }

    # Surrogate Input: Params + One-Hot Sublattice ID
    X_params = np.repeat(param_vector[None, :], num_sites, axis=0)
    
    # Create one-hot vectors for sublattices A, B, C (0, 1, 2)
    sublattice_ids = np.eye(3, dtype=np.float32)[np.arange(num_sites) % 3]

    # Surrogate Input: Every node gets the [U, t1, t2, theta_x, theta_y] parameter vector
    X_data = np.concatenate([X_params, sublattice_ids], axis=1)

    print(f"Loaded successfully! Lattice Sites: {num_sites} | (Lx={Lx}, Ly={Ly})")
    return X_data, observables, param_vector, num_sites, Lx, Ly


# ==========================================
# 3. GNN MODULES (SURROGATE)
# ==========================================
class GCNLayer(nn.Module):
    features: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        x = nn.Dense(self.features)(node_feats)
        x = jnp.matmul(adj_matrix, x)
        return nn.silu(x)


class NodePropertyHead(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_feats):
        x = nn.Dense(self.hidden_dim)(node_feats)
        x = nn.silu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


class BilinearCorrelationHead(nn.Module):
    @nn.compact
    def __call__(self, node_feats):
        feat_dim = node_feats.shape[-1]
        W = self.param('W', nn.initializers.normal(stddev=0.1), (feat_dim, feat_dim))
        raw_corr = jnp.einsum('bni,ij,bmj->bnm', node_feats, W, node_feats)
        return 0.5 * (raw_corr + jnp.swapaxes(raw_corr, 1, 2))


class GlobalPropertyHead(nn.Module):
    """Pools graph features to predict a single macroscopic scalar (e.g., Energy)."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_feats):
        global_feat = jnp.sum(node_feats, axis=1) 
        x = nn.Dense(self.hidden_dim)(global_feat)
        x = nn.silu(x)
        x = nn.Dense(1)(x)
        return x


class KagomeGNN_Surrogate(nn.Module):
    """Parameter-to-Observable Surrogate Model."""
    hidden_dim: int = 128
    head_dim: int = 64

    @nn.compact
    def __call__(self, x_input, adj_matrix):
        # x_input shape: (batch, num_sites, 5)
        h = GCNLayer(self.hidden_dim)(x_input, adj_matrix)
        h = GCNLayer(self.hidden_dim)(h, adj_matrix)

        pred_heads = {
            "charge_density": NodePropertyHead(hidden_dim=self.head_dim)(h),
            "spin_corr": BilinearCorrelationHead()(h),
            "energy": GlobalPropertyHead(hidden_dim=self.head_dim)(h)
        }

        return pred_heads


# ==========================================
# 4. TRAINING LOOP (SURROGATE)
# ==========================================
def train_kagome_surrogate(X_data, A_data, observables_true, params, opt_state, optimizer, model, num_epochs=600):
    print("--- Training Surrogate Parameter-to-Observable GNN ---")

    targets_jax = {k: jnp.array(v) for k, v in observables_true.items()}
    X_inputs = jnp.array(X_data)
    A_inputs = jnp.array(A_data)

    # Automatically compute loss weights based on the inverse variance of each target
    target_variances = {k: jnp.var(v) + 1e-6 for k, v in targets_jax.items()}
    loss_weights = {k: 1.0 / v for k, v in target_variances.items()}

    print("\n[Automatic Loss Weights (Inverse Variance)]:")
    for k, w in loss_weights.items():
        print(f"  - {k}: {float(w):.4f}")
    print()

    # Track loss history
    history = {
        "total": [],
        "charge_density": [],
        "spin_corr": [],
        "energy": []
    }

    @jax.jit
    def loss_fn(params, x_batch, adj_batch):
        pred_heads = model.apply({"params": params}, x_batch, adj_batch)

        total_obs_loss = 0.0
        head_losses = {}
        for name, target in targets_jax.items():
            l_head = jnp.mean((pred_heads[name] - target) ** 2)
            head_losses[name] = l_head
            
            # Apply our computed weights to balance the scales!
            total_obs_loss += l_head * loss_weights[name]

        return total_obs_loss, (head_losses, pred_heads)

    @jax.jit
    def train_step(params, opt_state, x_batch, adj_batch):
        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_batch, adj_batch
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, aux

    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state, X_inputs, A_inputs)
        head_losses, pred_heads = aux

        # Save history for plotting
        history["total"].append(float(total_loss))
        for k, v in head_losses.items():
            history[k].append(float(v))

        if (epoch + 1) % 150 == 0:
            loss_str = " | ".join([f"{k[:6]}: {float(v):.2e}" for k, v in head_losses.items()])
            print(f"Epoch {epoch+1:4d} | Weighted Total Loss: {float(total_loss):.4f} | {loss_str}")

    # Return full batched predictions and history
    pred_heads_numpy = {k: np.array(v) for k, v in pred_heads.items()}
    return params, opt_state, pred_heads_numpy, history


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # 1. ACCUMULATE ALL DATA FOR BATCHING
    print("Gathering dataset...")
    all_X, all_A = [], []
    all_obs = {"charge_density": [], "spin_corr": [], "energy": []}
    loaded_files = [] 

    for j in range(10):
        for i in range(100):
            h5_file = r"C:\Users\jj9ba\Research\DREU QIS-AI\kagome-datasets\kagome-datasets\wavefunctions\three_site_kagome_"+f"{i:03d}.h5"
            try:
                X_data, obs_true, param_vec, num_sites, Lx, Ly = load_kagome_dataset(h5_file)
                A_data = build_kagome_adjacency(Lx=Lx, Ly=Ly, periodic=True)

                all_X.append(X_data)
                all_A.append(A_data)
                for k in all_obs.keys():
                    all_obs[k].append(obs_true[k])
                loaded_files.append(i)
            except FileNotFoundError:
                continue

    batch_X = np.stack(all_X, axis=0)
    batch_A = np.stack(all_A, axis=0)
    batch_obs = {k: np.stack(v, axis=0) for k, v in all_obs.items()}

    print(f"\nDataset batched! Total files loaded: {batch_X.shape[0]}")
    print(f"Batched Input Shape: {batch_X.shape} -> (Batch, Nodes, Features)\n")

    # ==========================================
    # NEW: Z-SCORE NORMALIZATION
    # ==========================================
    print("Applying Z-score normalization...")
    
    # 1. Normalize Inputs (batch_X)
    X_mean = np.mean(batch_X, axis=(0, 1), keepdims=True)
    X_std = np.std(batch_X, axis=(0, 1), keepdims=True) + 1e-8
    batch_X_norm = (batch_X - X_mean) / X_std

    # 2. Normalize Targets (batch_obs)
    obs_stats = {}
    batch_obs_norm = {}
    for k, v in batch_obs.items():
        # Compute mean and std across the batch dimension
        mean_val = np.mean(v, axis=0, keepdims=True)
        std_val = np.std(v, axis=0, keepdims=True) + 1e-8
        obs_stats[k] = {'mean': mean_val, 'std': std_val}
        
        # Create normalized target arrays
        batch_obs_norm[k] = (v - mean_val) / std_val
    # ==========================================

    # 2. INITIALIZE MODEL
    model = KagomeGNN_Surrogate(hidden_dim=128, head_dim=64)
    rng = jax.random.PRNGKey(42)

    dummy_x = jnp.ones((1, 3, 8)) 
    dummy_a = jnp.ones((1, 3, 3))
    params = model.init(rng, dummy_x, dummy_a)["params"]

    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=80000, alpha=0.01)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # 3. TRAIN ON THE NORMALIZED DATASET
    # Notice we pass batch_X_norm and batch_obs_norm here and catch history
    params, opt_state, pred_heads_norm, history = train_kagome_surrogate(
        batch_X_norm, batch_A, batch_obs_norm, params, opt_state, optimizer, model, num_epochs=60
    )

    # ==========================================
    # NEW: PLOT LOSS CURVES
    # ==========================================
    print("\nPlotting loss curves...")
    epochs_range = range(1, len(history["total"]) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Total Weighted Loss (thicker line)
    plt.plot(epochs_range, history["total"], label='Weighted Total Loss', 
             color='black', linewidth=2, linestyle='--')
             
    # Plot individual observable losses
    plt.plot(epochs_range, history["energy"], label='Energy Loss')
    plt.plot(epochs_range, history["charge_density"], label='Charge Density Loss')
    plt.plot(epochs_range, history["spin_corr"], label='Spin Corr Loss')

    plt.yscale('log') # Log scale is best for viewing MSE convergence
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (Log Scale)')
    plt.title('Training Loss per Observable')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save the plot and show it
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=300)
    plt.show()

    # ==========================================
    # NEW: UN-NORMALIZE PREDICTIONS
    # ==========================================
    # We must scale the predictions back up to real physical units for printing
    pred_heads = {}
    for k in pred_heads_norm.keys():
        mean_val = np.squeeze(obs_stats[k]['mean'])
        std_val = np.squeeze(obs_stats[k]['std'])
        pred_heads[k] = (pred_heads_norm[k] * std_val) + mean_val
    # ==========================================

    # 4. DISPLAY RESULTS
    print("\nRESULTS PER FILE (True vs Predicted in Physical Units)")
    print("=======================================================")
    
    num_files = batch_X.shape[0]
    for i in range(num_files):
        file_idx = loaded_files[i]
        print(f"--- File three_site_kagome_{file_idx:03d}.h5 ---")
        
        # Safely extract and cast to float
        e_true = float(np.squeeze(batch_obs['energy'][i]))
        e_pred = float(np.squeeze(pred_heads['energy'][i]))
        print(f"  Energy:        True = {e_true:8.4f} | Pred = {e_pred:8.4f}")
        
        c_true = float(np.squeeze(batch_obs['charge_density'][i][0]))
        c_pred = float(np.squeeze(pred_heads['charge_density'][i][0]))
        print(f"  Charge (site0): True = {c_true:8.4f} | Pred = {c_pred:8.4f}")
        
        print("  Spin Matrix (True):")
        for row in batch_obs['spin_corr'][i]:
            print("    " + " ".join([f"{val:8.4f}" for val in row]))
            
        print("  Spin Matrix (Pred):")
        for row in pred_heads['spin_corr'][i]:
            print("    " + " ".join([f"{val:8.4f}" for val in row]))
            
        print("-" * 55)