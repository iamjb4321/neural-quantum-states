import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jraph

# ==========================================
# 1. KAGOME EDGE GENERATOR (For Jraph)
# ==========================================
def build_kagome_edges(Lx: int, Ly: int, periodic: bool = True):
    """Builds senders and receivers arrays for Jraph."""
    num_unit_cells = Lx * Ly
    num_sites = 3 * num_unit_cells
    
    senders, receivers = [], []

    def get_site_index(x, y, sublattice):
        if periodic:
            x, y = x % Lx, y % Ly
        elif x < 0 or x >= Lx or y < 0 or y >= Ly:
            return -1
        return 3 * (x + y * Lx) + sublattice

    def add_edge(site1, site2):
        if site1 != -1 and site2 != -1:
            senders.extend([site1, site2])
            receivers.extend([site2, site1])

    for y in range(Ly):
        for x in range(Lx):
            A, B, C = get_site_index(x, y, 0), get_site_index(x, y, 1), get_site_index(x, y, 2)
            # Intra-unit cell
            add_edge(A, B); add_edge(B, C); add_edge(C, A)
            # Inter-unit cell
            add_edge(A, get_site_index(x - 1, y, 1))
            add_edge(C, get_site_index(x, y - 1, 0))
            add_edge(B, get_site_index(x - 1, y - 1, 2))

    return jnp.array(senders), jnp.array(receivers)


# ==========================================
# 2. HDF5 DATASET LOADER
# ==========================================
def load_kagome_dataset(h5_filepath):
    if not os.path.exists(h5_filepath):
        raise FileNotFoundError(f"Could not locate file: {h5_filepath}")

    with h5py.File(h5_filepath, "r") as f:
        p = f["model_params"]
        Lx = int(p["Lx"][()]) if "Lx" in p else 1
        Ly = int(p["Ly"][()]) if "Ly" in p else 1
        U = float(p["U"][()]) if "U" in p else 0.0
        t1 = float(p["t1"][()]) if "t1" in p else 1.0
        t2 = float(p["t2"][()]) if "t2" in p else 0.0
        theta_x, theta_y = (float(p.get(k, 0.0)[()]) for k in ("theta_x", "theta_y"))

        param_vector = np.array([U, t1, t2, theta_x, theta_y], dtype=np.float32)
        
        density_true = np.squeeze(np.array(f["observables"]["charge_density"], dtype=np.float32))
        spin_corr_true = np.squeeze(np.array(f["observables"]["spin_corr"], dtype=np.float32))
        energy_true = float(f["energy"][()]) if "energy" in f else 0.0

        num_sites = len(density_true)

    X_params = np.repeat(param_vector[None, :], num_sites, axis=0)
    sublattice_ids = np.eye(3, dtype=np.float32)[np.arange(num_sites) % 3]
    X_data = np.concatenate([X_params, sublattice_ids], axis=1)

    return X_data, {"charge_density": density_true, "spin_corr": spin_corr_true, "energy": np.array([energy_true], dtype=np.float32)}, num_sites, Lx, Ly


# ==========================================
# 3. THE SIZE-AGNOSTIC JRAPH MODEL
# ==========================================
class PairwiseSpinCorrelation(nn.Module):
    """Size-agnostic N x N matrix generator using JAX vmap."""
    hidden_dim: int

    @nn.compact
    def __call__(self, node_embeddings):
        def predict_single_pair(h_i, h_j):
            h_pair = jnp.concatenate([h_i, h_j], axis=-1)
            x = nn.silu(nn.Dense(self.hidden_dim)(h_pair))
            return nn.Dense(1)(x)[0]
            
        row_fn = jax.vmap(predict_single_pair, in_axes=(None, 0))
        matrix_fn = jax.vmap(row_fn, in_axes=(0, None))
        return matrix_fn(node_embeddings, node_embeddings)


class KagomeGNN_Surrogate(nn.Module):
    hidden_dim: int = 128
    head_dim: int = 64

    @nn.compact
    def __call__(self, nodes, senders, receivers):
        # 1. Message Passing - Fixes the NoneType Error!
        # Nodes send their features. Receivers sum them up and concatenate with their own state.
        def update_edge_fn(edges, senders, receivers, globals_):
            return nn.Dense(self.hidden_dim)(senders)

        def update_node_fn(nodes, sent_edges, received_edges, globals_):
            x = jnp.concatenate([nodes, received_edges], axis=-1)
            return nn.silu(nn.Dense(self.hidden_dim)(x))

        gnn = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn)

        graph = jraph.GraphsTuple(
            nodes=nodes, edges=None, senders=senders, receivers=receivers,
            n_node=jnp.array([nodes.shape[0]]), n_edge=jnp.array([senders.shape[0]]), globals=None
        )

        # Apply layers
        graph = gnn(graph)
        graph = gnn(graph)
        h = graph.nodes # Extracted rich features

        # 2. Local Head (Charge)
        charge = jnp.squeeze(nn.Sequential([nn.Dense(self.head_dim), nn.silu, nn.Dense(1)])(h), axis=-1)

        # 3. Global Head (Energy) - Size Agnostic Pooling
        energy = nn.Sequential([nn.Dense(self.head_dim), nn.silu, nn.Dense(1)])(jnp.mean(h, axis=0))

        # 4. Pairwise Head (Spin)
        spin_corr = PairwiseSpinCorrelation(self.head_dim)(h)

        return {"charge_density": charge, "spin_corr": spin_corr, "energy": energy}


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_kagome_surrogate(X_data, S_data, R_data, observables_true, params, opt_state, optimizer, model, num_epochs=600):
    print("--- Training Surrogate Parameter-to-Observable GNN ---")

    targets_jax = {k: jnp.array(v) for k, v in observables_true.items()}
    loss_weights = {k: 1.0 / (jnp.var(v) + 1e-6) for k, v in targets_jax.items()}

    @jax.jit
    def train_step(params, opt_state, x_batch, s_batch, r_batch):
        def loss_fn(p):
            # vmap maps the single-graph model across the batch dimension
            pred_heads = jax.vmap(model.apply, in_axes=({"params": None}, 0, 0, 0))({"params": p}, x_batch, s_batch, r_batch)
            head_losses = {name: jnp.mean((pred_heads[name] - targets_jax[name]) ** 2) for name in targets_jax}
            total_loss = sum(head_losses[name] * loss_weights[name] for name in targets_jax)
            return total_loss, (head_losses, pred_heads)

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, total_loss, aux

    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state, X_data, S_data, R_data)
        head_losses, pred_heads = aux

        if (epoch + 1) % 150 == 0:
            loss_str = " | ".join([f"{k[:6]}: {float(v):.2e}" for k, v in head_losses.items()])
            print(f"Epoch {epoch+1:4d} | Weighted Total Loss: {float(total_loss):.4f} | {loss_str}")

    return params, {k: np.array(v) for k, v in pred_heads.items()}


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("Gathering dataset...")
    all_X, all_S, all_R, all_obs, loaded_files = [], [], [], {"charge_density": [], "spin_corr": [], "energy": []}, []

    for i in range(100):
        h5_file = r"C:\Users\jj9ba\Research\DREU QIS-AI\kagome-datasets\kagome-datasets\wavefunctions\three_site_kagome_"+f"{i:03d}.h5"
        try:
            X_data, obs_true, num_sites, Lx, Ly = load_kagome_dataset(h5_file)
            S_data, R_data = build_kagome_edges(Lx=Lx, Ly=Ly, periodic=True)

            all_X.append(X_data); all_S.append(S_data); all_R.append(R_data)
            for k in all_obs: all_obs[k].append(obs_true[k])
            loaded_files.append(i)
        except FileNotFoundError:
            continue

    batch_X = jnp.array(all_X)
    batch_S = jnp.array(all_S)
    batch_R = jnp.array(all_R)
    batch_obs = {k: np.array(v) for k, v in all_obs.items()}

    print(f"\nDataset batched! Total files loaded: {batch_X.shape[0]}")

    # Normalization
    X_mean, X_std = np.mean(batch_X, axis=(0, 1), keepdims=True), np.std(batch_X, axis=(0, 1), keepdims=True) + 1e-8
    batch_X_norm = (batch_X - X_mean) / X_std

    obs_stats, batch_obs_norm = {}, {}
    for k, v in batch_obs.items():
        m, s = np.mean(v, axis=0, keepdims=True), np.std(v, axis=0, keepdims=True) + 1e-8
        obs_stats[k], batch_obs_norm[k] = {'mean': m, 'std': s}, (v - m) / s

    # Model Initialization (NO DUMMY DATA! We use the first graph in the batch)
    model = KagomeGNN_Surrogate(hidden_dim=128, head_dim=64)
    params = model.init(jax.random.PRNGKey(42), batch_X_norm[0], batch_S[0], batch_R[0])["params"]

    optimizer = optax.adam(learning_rate=optax.cosine_decay_schedule(1e-3, 1500, 0.01))
    opt_state = optimizer.init(params)

    # Train
    params, pred_heads_norm = train_kagome_surrogate(
        batch_X_norm, batch_S, batch_R, batch_obs_norm, params, opt_state, optimizer, model, num_epochs=150
    )

    # Un-Normalize
    pred_heads = {k: (pred_heads_norm[k] * np.squeeze(obs_stats[k]['std'])) + np.squeeze(obs_stats[k]['mean']) for k in pred_heads_norm}

    # Print Results
    print("\nRESULTS PER FILE (True vs Predicted in Physical Units)")
    print("=======================================================")
    for i, file_idx in enumerate(loaded_files):
        print(f"--- File three_site_kagome_{file_idx:03d}.h5 ---")
        
        # Add [0] to the energy arrays to extract the scalar value!
        print(f"  Energy:        True = {float(batch_obs['energy'][i][0]):8.4f} | Pred = {float(pred_heads['energy'][i][0]):8.4f}")
        
        print(f"  Charge (site0): True = {float(batch_obs['charge_density'][i][0]):8.4f} | Pred = {float(pred_heads['charge_density'][i][0]):8.4f}")
        
        print("  Spin Matrix (True):")
        for row in batch_obs['spin_corr'][i]: print("    " + " ".join([f"{val:8.4f}" for val in row]))
            
        print("  Spin Matrix (Pred):")
        for row in pred_heads['spin_corr'][i]: print("    " + " ".join([f"{val:8.4f}" for val in row]))
        print("-" * 55)