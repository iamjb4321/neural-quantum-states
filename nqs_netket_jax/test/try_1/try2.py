import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# ==========================================
# 1. BIPARTITE KAGOME ADJACENCY GENERATOR
# ==========================================
def build_bipartite_kagome_adjacency(Lx: int, Ly: int, periodic: bool = True):
    """
    Builds bipartite adjacency matrices connecting Spins to Triangles (Plaquettes).
    Returns adj_spin_to_plaq of shape (num_plaquettes, num_sites).
    """
    num_unit_cells = Lx * Ly
    num_sites = 3 * num_unit_cells
    num_plaq = 2 * num_unit_cells # Up-triangles and Down-triangles
    
    # Adjacency: 1 if site is in plaquette, 0 otherwise
    adj_sp = np.zeros((num_plaq, num_sites), dtype=np.float32)

    def get_site(x, y, sub):
        if periodic:
            x, y = x % Lx, y % Ly
        elif x < 0 or x >= Lx or y < 0 or y >= Ly:
            return -1
        return 3 * (x + y * Lx) + sub

    for y in range(Ly):
        for x in range(Lx):
            # Up-Triangle (A, B, C in the same unit cell)
            p_up = 2 * (x + y * Lx)
            sA = get_site(x, y, 0)
            sB = get_site(x, y, 1)
            sC = get_site(x, y, 2)
            for s in [sA, sB, sC]:
                if s != -1: adj_sp[p_up, s] = 1.0
                
            # Down-Triangle (C_local, A_down, B_left)
            p_down = 2 * (x + y * Lx) + 1
            sC_loc = get_site(x, y, 2)
            sA_dn  = get_site(x, y - 1, 0)
            sB_lf  = get_site(x - 1, y, 1)
            for s in [sC_loc, sA_dn, sB_lf]:
                if s != -1: adj_sp[p_down, s] = 1.0

    return jnp.array(adj_sp)


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

        param_vector = np.array([
            float(p.get("U", 0.0)[()]), 
            float(p.get("t1", 1.0)[()]), 
            float(p.get("t2", 0.0)[()]), 
            float(p.get("theta_x", 0.0)[()]), 
            float(p.get("theta_y", 0.0)[()])
        ], dtype=np.float32)

        obs = f["observables"]
        
        observables = {}
        observables["charge_density"] = np.squeeze(np.array(obs["charge_density"], dtype=np.float32))
        observables["spin_corr"] = np.squeeze(np.array(obs["spin_corr"], dtype=np.float32))
        
        if "charge_corr" in obs:
            observables["charge_corr"] = np.squeeze(np.array(obs["charge_corr"], dtype=np.float32))
        else:
            observables["charge_corr"] = np.zeros_like(observables["spin_corr"]) 
            
        if "double_occupancy" in obs:
            observables["double_occupancy"] = np.squeeze(np.array(obs["double_occupancy"], dtype=np.float32))
        else:
            observables["double_occupancy"] = np.zeros_like(observables["charge_density"]) 

        if "energy" in f:
            observables["energy"] = np.array([float(f["energy"][()])], dtype=np.float32)
        else:
            observables["energy"] = np.array([0.0], dtype=np.float32)

        num_sites = len(observables["charge_density"])

    # Surrogate Input: Params + One-Hot Sublattice ID
    X_params = np.repeat(param_vector[None, :], num_sites, axis=0)
    sublattice_ids = np.eye(3, dtype=np.float32)[np.arange(num_sites) % 3]
    X_data = np.concatenate([X_params, sublattice_ids], axis=1)

    return X_data, observables, param_vector, num_sites, Lx, Ly


# ==========================================
# 3. GAT & PLAQUETTE MODULES
# ==========================================
class BipartiteGATLayer(nn.Module):
    """Attention-based message passing between sets of nodes."""
    features: int

    @nn.compact
    def __call__(self, src_feats, dst_feats_init, adj_matrix):
        h_src = nn.Dense(self.features)(src_feats)
        
        if dst_feats_init is None:
            B = src_feats.shape[0]
            N_dst = adj_matrix.shape[0]
            h_dst = jnp.zeros((B, N_dst, self.features))
        else:
            h_dst = nn.Dense(self.features)(dst_feats_init)

        W_attn = self.param('W_attn', nn.initializers.normal(0.1), (self.features, self.features))
        scores = jnp.einsum('bdi,ij,bsj->bds', h_dst, W_attn, h_src)
        
        mask = jnp.where(adj_matrix > 0, 0.0, -1e9)
        attn_weights = nn.softmax(scores + mask, axis=-1)
        
        out = jnp.einsum('bds,bsf->bdf', attn_weights, h_src)
        return nn.silu(out)


# ==========================================
# 4. PREDICTION HEADS
# ==========================================
class NodePropertyHead(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_feats):
        x = nn.Dense(self.hidden_dim)(node_feats)
        x = nn.silu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)

class PairwiseCorrelationHead(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, h):
        B, N, F = h.shape
        h_i = jnp.tile(jnp.expand_dims(h, 2), (1, 1, N, 1))
        h_j = jnp.tile(jnp.expand_dims(h, 1), (1, N, 1, 1))
        pair_h = jnp.concatenate([h_i, h_j], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(pair_h)
        x = nn.silu(x)
        out = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        
        return 0.5 * (out + jnp.swapaxes(out, 1, 2))

class ExtensiveGlobalHead(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_feats, num_sites):
        global_feat = jnp.mean(node_feats, axis=1) 
        x = nn.Dense(self.hidden_dim)(global_feat)
        x = nn.silu(x)
        energy_density = nn.Dense(1)(x)
        return energy_density * num_sites


class KagomeHubbardGNN(nn.Module):
    """The Complete Architecture with Decoupled Spin Branch and Uncertainty Weighting."""
    hidden_dim: int = 128
    head_dim: int = 64

    @nn.compact
    def __call__(self, x_spins, adj_spin_to_plaq, num_sites):
        adj_plaq_to_spin = adj_spin_to_plaq.T

        # --- PLAQUETTE MESSAGE PASSING BACKBONE ---
        h_plaq = BipartiteGATLayer(self.hidden_dim)(
            src_feats=x_spins, dst_feats_init=None, adj_matrix=adj_spin_to_plaq
        )
        h_spins_updated = BipartiteGATLayer(self.hidden_dim)(
            src_feats=h_plaq, dst_feats_init=x_spins, adj_matrix=adj_plaq_to_spin
        )
        h_plaq = BipartiteGATLayer(self.hidden_dim)(
            src_feats=h_spins_updated, dst_feats_init=h_plaq, adj_matrix=adj_spin_to_plaq
        )
        final_spins = BipartiteGATLayer(self.hidden_dim)(
            src_feats=h_plaq, dst_feats_init=h_spins_updated, adj_matrix=adj_plaq_to_spin
        )

        # --- DECOUPLE MAGNETIC FEATURES ---
        spin_feats = nn.Dense(self.hidden_dim)(final_spins)
        spin_feats = nn.silu(spin_feats)
        spin_feats = nn.Dense(self.hidden_dim)(spin_feats)
        spin_feats = nn.silu(spin_feats)

        # --- DYNAMIC LOSS WEIGHTS (Uncertainty Weighting) ---
        log_vars = {
            "charge_density": self.param('lv_charge', nn.initializers.zeros, ()),
            "double_occupancy": self.param('lv_docc', nn.initializers.zeros, ()),
            "spin_corr": self.param('lv_spin', nn.initializers.zeros, ()),
            "charge_corr": self.param('lv_ccorr', nn.initializers.zeros, ()),
            "energy": self.param('lv_energy', nn.initializers.zeros, ())
        }

        # --- MULTI-TASK PREDICTION HEADS ---
        pred_heads = {
            "charge_density":   NodePropertyHead(self.head_dim)(final_spins),
            "double_occupancy": NodePropertyHead(self.head_dim)(final_spins),
            "charge_corr":      PairwiseCorrelationHead(self.head_dim)(final_spins),
            "energy":           ExtensiveGlobalHead(self.head_dim)(final_spins, num_sites),
            "spin_corr":        PairwiseCorrelationHead(self.head_dim)(spin_feats) # Use decoupled feats
        }
        
        return pred_heads, log_vars


# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_kagome_surrogate(X_data, A_data, num_sites_array, observables_true, params, opt_state, optimizer, model, num_epochs=600):
    targets_jax = {k: jnp.array(v) for k, v in observables_true.items()}
    X_inputs = jnp.array(X_data)
    A_inputs = jnp.array(A_data)
    num_sites_jnp = jnp.array(num_sites_array).reshape(-1, 1)

    @jax.jit
    def loss_fn(params, x_batch, adj_batch, ns_batch):
        pred_heads, log_vars = model.apply({"params": params}, x_batch, adj_batch[0], ns_batch)

        total_obs_loss = 0.0
        head_losses = {}
        
        for name, target in targets_jax.items():
            l_head = jnp.mean((pred_heads[name] - target) ** 2)
            head_losses[name] = l_head
            
            # Uncertainty Weighting formula
            precision = jnp.exp(-log_vars[name])
            total_obs_loss += 0.5 * precision * l_head + 0.5 * log_vars[name]

        return total_obs_loss, (head_losses, pred_heads)

    @jax.jit
    def train_step(params, opt_state, x_batch, adj_batch, ns_batch):
        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_batch, adj_batch, ns_batch
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, aux

    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state, X_inputs, A_inputs, num_sites_jnp)
        head_losses, pred_heads = aux

        if (epoch + 1) % 150 == 0:
            loss_str = " | ".join([f"{k[:6]}: {float(v):.2e}" for k, v in head_losses.items()])
            print(f"Epoch {epoch+1:4d} | Weighted Loss: {float(total_loss):.4f} | {loss_str}")

    pred_heads_numpy = {k: np.array(v) for k, v in pred_heads.items()}
    return params, opt_state, pred_heads_numpy, head_losses


# ==========================================
# 6. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("Gathering dataset...")
    all_X, all_A, all_Nsites = [], [], []
    all_obs = {"charge_density": [], "spin_corr": [], "charge_corr": [], "double_occupancy": [], "energy": []}
    loaded_files = [] 

    for i in range(100):
        h5_file = r"C:\Users\jj9ba\Research\DREU QIS-AI\kagome-datasets\kagome-datasets\wavefunctions\three_site_kagome_"+f"{i:03d}.h5"
        try:
            X_data, obs_true, param_vec, num_sites, Lx, Ly = load_kagome_dataset(h5_file)
            A_sp_plaq = build_bipartite_kagome_adjacency(Lx=Lx, Ly=Ly, periodic=True)

            all_X.append(X_data)
            all_A.append(A_sp_plaq)
            all_Nsites.append(num_sites)
            for k in all_obs.keys():
                all_obs[k].append(obs_true[k])
            loaded_files.append(i)
        except FileNotFoundError:
            continue

    if len(all_X) == 0:
        print("No files loaded. Creating mock data to demonstrate execution...")
        Lx, Ly = 2, 2
        num_sites = 3 * Lx * Ly
        all_X = [np.random.randn(num_sites, 8) for _ in range(2)]
        all_A = [build_bipartite_kagome_adjacency(Lx, Ly) for _ in range(2)]
        all_Nsites = [num_sites, num_sites]
        for k in all_obs.keys():
            if "corr" in k: all_obs[k] = [np.random.randn(num_sites, num_sites) for _ in range(2)]
            elif "energy" in k: all_obs[k] = [np.random.randn(1) for _ in range(2)]
            else: all_obs[k] = [np.random.randn(num_sites) for _ in range(2)]
        loaded_files = [0, 1]

    batch_X = np.stack(all_X, axis=0)
    batch_A = np.stack(all_A, axis=0)
    batch_Nsites = np.array(all_Nsites)
    batch_obs = {k: np.stack(v, axis=0) for k, v in all_obs.items()}

    print(f"\nDataset batched! Total files loaded: {batch_X.shape[0]}")

    # Normalize Inputs
    X_mean, X_std = np.mean(batch_X, axis=(0, 1), keepdims=True), np.std(batch_X, axis=(0, 1), keepdims=True) + 1e-8
    batch_X_norm = (batch_X - X_mean) / X_std

    # Normalize Targets using Global Scalar Normalization
    obs_stats, batch_obs_norm = {}, {}
    for k, v in batch_obs.items():
        mean_val = np.mean(v)
        std_val = np.std(v) + 1e-8
        
        obs_stats[k] = {'mean': mean_val, 'std': std_val}
        batch_obs_norm[k] = (v - mean_val) / std_val

    # Initialize Model
    model = KagomeHubbardGNN(hidden_dim=128, head_dim=64)
    rng = jax.random.PRNGKey(42)
    
    # Dummy pass for init
    dummy_x = jnp.ones_like(batch_X_norm[:1])
    dummy_a = jnp.ones_like(batch_A[0])
    params = model.init(rng, dummy_x, dummy_a, num_sites=batch_Nsites[0])["params"]

    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=1500, alpha=0.01)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # Train
    params, opt_state, pred_heads_norm, head_losses = train_kagome_surrogate(
        batch_X_norm, batch_A, batch_Nsites, batch_obs, params, opt_state, optimizer, model, num_epochs=1500
    ) # changed batch_obs_norm -----> batch_obs

    # Un-normalize Predictions
    pred_heads = {}
    for k in pred_heads_norm.keys():
        mean_val = obs_stats[k]['mean']
        std_val = obs_stats[k]['std']
        pred_heads[k] = (pred_heads_norm[k] * std_val) + mean_val

    # 4. DISPLAY RESULTS
    print("\nRESULTS PER FILE (True vs Predicted in Physical Units)")
    print("=======================================================")
    for i in range(len(loaded_files)):
        file_idx = loaded_files[i]
        print(f"--- File three_site_kagome_{file_idx:03d}.h5 ---")
        
        for k in ["charge_density", "double_occupancy"]:

            v_true = np.asarray(batch_obs[k][i]).flatten()
            v_pred = np.asarray(pred_heads[k][i]).flatten()
            
            label = f"  {k.replace('_', ' ').title()}:".ljust(22)
            formatted_string = f"{label} True = {', '.join(f'{x:8.2f}' for x in v_true)} | Pred = {', '.join(f'{x:8.2f}' for x in v_pred)}"
            print(formatted_string)

        k = "energy"
        v_true = float(np.asarray(batch_obs["energy"][i]).flatten()[0])
        v_pred = float(np.asarray(pred_heads["energy"][i]).flatten()[0])

        # Formatting a list of floats to 2 decimal places 
        # numbers = [1.2345, 6.7891, 0.5555] 
        # formatted_string = f"My list: {', '.join(f'{x:.2f}' for x in numbers)}" 
        # print(formatted_string) 
        # # Output: My list: 1.23, 6.79, 0.56


        
        label = f"  {k.replace('_', ' ').title()}:".ljust(22)
        print(f"{label} True = {v_true:8.4f}  | Pred = {v_pred:8.4f}")
        print("  Spin Corr Matrix [Row 0] (True):  " + " ".join([f"{val:6.3f}" for val in batch_obs['spin_corr'][i][0][:5]]))
        print("  Spin Corr Matrix [Row 0] (Pred):  " + " ".join([f"{val:6.3f}" for val in pred_heads['spin_corr'][i][0][:5]]))
        print("  Charge Corr Matrix [R0] (True):   " + " ".join([f"{val:6.3f}" for val in batch_obs['charge_corr'][i][0][:5]]))
        print("-" * 65)