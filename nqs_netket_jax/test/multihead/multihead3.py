import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# Try importing TeNPy helper for native MPS deserialization if available
try:
    import tenpy.tools.hdf5 as h5_tenpy
    from tenpy.algorithms.exact_diag import get_full_wavefunction
    HAS_TENPY = True
except ImportError:
    HAS_TENPY = False


# ==========================================
# 1. HDF5 DATASET LOADER FOR KAGOME H5
# ==========================================
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
        double_occupancy_true = np.squeeze(get_obs("double_occupancy"))
        s_q_charge_true = np.squeeze(get_obs("S_q_charge"))
        S_q_spin_true = np.squeeze(get_obs("S_q_spin"))
        spin_corr_true = np.squeeze(get_obs("spin_corr"))
        charge_corr_true = np.squeeze(get_obs("charge_corr"))

        num_sites = len(density_true)

        observables = {
            "charge_density": density_true,
            "double_occupancy": double_occupancy_true,
            "s_q_charge": s_q_charge_true,
            "S_q_spin": S_q_spin_true,
            "spin_corr": spin_corr_true,
            "charge_corr": charge_corr_true,
        }

        # --- C. Read or Deserialise Wavefunction ---
        y_psi = None
        if "psi" in f and HAS_TENPY:
            print("Deserializing native TeNPy MPS object from HDF5...")
            psi_mps = h5_tenpy.load_from_hdf5(f, "psi")
            psi_dense = get_full_wavefunction(psi_mps).to_ndarray()
            y_psi = psi_dense.reshape(-1).astype(np.float32)
        elif "y_psi" in f:
            y_psi = np.array(f["y_psi"], dtype=np.float32)

    # --- D. Build Basis Matrix X if State Vector exists ---
    if y_psi is not None:
        dim = len(y_psi)
        y_psi = y_psi / np.linalg.norm(y_psi)

        basis_indices = np.arange(dim)
        binary_matrix = (basis_indices[:, None] >> np.arange(num_sites)[::-1]) & 1
        X_data = (2 * binary_matrix - 1).astype(np.float32)
    else:
        print("Note: No explicit state vector found in file. Training on observables directly.")
        X_data = None

    print(f"Loaded successfully!")
    print(f"Lattice Sites: {num_sites} | Params Vector: {param_vector}")
    print("Observables registered:")
    for k, v in observables.items():
        print(f"  - {k}: shape {v.shape}")
    print()

    return X_data, y_psi, observables, param_vector, num_sites


# ==========================================
# 2. SUB-HEAD MODULES (1D VECTOR & 2D SYMMETRIC)
# ==========================================
class PropertySubHead(nn.Module):
    """Multi-layer MLP sub-network for 1D vector observables (density, S(q))."""
    target_shape: tuple
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, h_global):
        x = nn.Dense(self.hidden_dim)(h_global)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        flat_dim = int(np.prod(self.target_shape))
        out_flat = nn.Dense(flat_dim)(x)
        out = jnp.reshape(out_flat, self.target_shape)
        return jnp.squeeze(out) if len(self.target_shape) == 1 else out


class CorrelationHead(nn.Module):
    """Dedicated sub-network for 2D matrix correlations that enforces physical symmetry C_ij = C_ji."""
    num_sites: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, h_global):
        x = nn.Dense(self.hidden_dim)(h_global)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        
        # Linear layer mapping to N x N matrix entries
        raw_mat = nn.Dense(self.num_sites * self.num_sites)(x)
        mat = jnp.reshape(raw_mat, (self.num_sites, self.num_sites))
        
        # Enforce exact matrix symmetry: 0.5 * (M + M^T)
        return 0.5 * (mat + mat.T)


class KagomePropertyNQS(nn.Module):
    num_sites: int
    obs_shapes: dict
    hidden_dim: int = 128
    head_dim: int = 64

    @nn.compact
    def __call__(self, x_input):
        # Shared Feature Extractor Trunk
        h = nn.Dense(self.hidden_dim)(x_input)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.tanh(h)

        # Head 0: Wavefunction Amplitudes log|psi(sigma)|
        log_psi = jnp.squeeze(nn.Dense(1)(h), axis=-1)

        # Global Pooled Representation for Property Prediction
        h_global = jnp.mean(h, axis=0, keepdims=True)

        # Multi-Layer Heads routed by tensor dimensionality
        pred_heads = {}
        for name, shape in self.obs_shapes.items():
            if len(shape) == 2 and shape[0] == shape[1]:
                # Enforce physical symmetry for N x N matrix correlation heads
                pred_heads[name] = CorrelationHead(
                    num_sites=shape[0],
                    hidden_dim=self.head_dim,
                    name=f"head_{name}"
                )(h_global)
            else:
                # Standard sub-head for 1D site or momentum vectors
                pred_heads[name] = PropertySubHead(
                    target_shape=shape,
                    hidden_dim=self.head_dim,
                    name=f"head_{name}"
                )(h_global)

        return log_psi, pred_heads


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_kagome_nqs(
    X_data, y_psi, observables_true, param_vector,
    num_epochs=1500, lr=1e-3, w_psi=1.0, obs_weights=None, model
):
    print("--- Training Multi-Head Neural Network with Symmetric Correlation Heads ---")

    obs_shapes = {k: v.shape for k, v in observables_true.items()}
    
    if obs_weights is None:
        obs_weights = {k: 0.5 for k in observables_true.keys()}


    # Cosine learning rate decay to prevent late-stage oscillations
    # Cosine learning rate decay (alpha = end_value / init_value = 1e-5 / 1e-3 = 0.01)
    schedule = optax.cosine_decay_schedule(
        init_value=lr, 
        decay_steps=num_epochs, 
        alpha=0.01
    )
    optimizer = optax.adam(learning_rate=schedule)

    targets_jax = {k: jnp.array(v) for k, v in observables_true.items()}

    has_psi = X_data is not None and y_psi is not None
    if has_psi:
        X_inputs = jnp.array(X_data)
        y_targets = jnp.array(y_psi)
    else:
        X_inputs = jnp.repeat(jnp.array(param_vector)[None, :], 16, axis=0)
        y_targets = None

    @jax.jit
    def loss_fn(params, x_batch, y_batch):
        log_psi, pred_heads = model.apply({"params": params}, x_batch)

        if has_psi:
            log_psi_shifted = log_psi - jnp.max(log_psi)
            psi = jnp.exp(log_psi_shifted)
            overlap = jnp.dot(psi, y_batch)
            norm_model = jnp.linalg.norm(psi)
            norm_true = jnp.linalg.norm(y_batch)
            loss_psi = 1.0 - ((overlap**2) / (norm_model**2 * norm_true**2 + 1e-12))
        else:
            loss_psi = 0.0

        total_obs_loss = 0.0
        head_losses = {}
        for name, target in targets_jax.items():
            l_head = jnp.mean((pred_heads[name] - target) ** 2)
            head_losses[name] = l_head
            total_obs_loss = total_obs_loss + obs_weights[name] * l_head

        total_loss = w_psi * loss_psi + total_obs_loss
        return total_loss, (loss_psi, head_losses, pred_heads)

    @jax.jit
    def train_step(params, opt_state):
        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, X_inputs, y_targets
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, aux

    rng = jax.random.PRNGKey(42)
    params = model.init(rng, X_inputs[:10])["params"]
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state)
        loss_psi, head_losses, pred_heads = aux

        if (epoch + 1) % 150 == 0:
            loss_str = " | ".join([f"{k[:6]}: {v:.2e}" for k, v in head_losses.items()])
            print(f"Epoch {epoch+1:4d} | Total Loss: {total_loss:.6f} | {loss_str}")

    print("\nTraining completed successfully!")
    return observables_true, pred_heads, head_losses


# ==========================================
# 4. RESULTS DISPLAY
# ==========================================
def display_results(observables_true, pred_heads, head_losses, num_sites):
    print("\n" + "=" * 65)
    print("      MULTI-HEAD NQS PHYSICAL PROPERTY PREDICTIONS")
    print("=" * 65)

    print("\n[Final Mean Squared Errors per Head]:")
    for name, loss_val in head_losses.items():
        print(f"  - Head '{name}': MSE = {float(loss_val):.4e}")

    if "charge_density" in observables_true:
        dens_true = observables_true["charge_density"]
        dens_pred = np.array(pred_heads["charge_density"])
        print("\n[Local Charge Density Comparison]:")
        print(f"  Site i  | H5 True Value | NQS Head Pred | Abs Error")
        print("  ----------------------------------------------------")
        for i in range(num_sites):
            err = abs(dens_true[i] - dens_pred[i])
            print(f"    {i:2d}    |   {dens_true[i]:+.5f}   |   {dens_pred[i]:+.5f}   |  {err:.2e}")

    if "spin_corr" in observables_true:
        corr_true = observables_true["spin_corr"]
        corr_pred = np.array(pred_heads["spin_corr"])
        print("\n[Spin Correlation Matrix Comparison (First 3x3 Block)]:")
        print("--- True H5 Spin Correlations ---")
        print(np.array2string(corr_true[:3, :3], precision=4, suppress_small=True))
        print("--- NQS Predicted Spin Correlations (Symmetrized) ---")
        print(np.array2string(corr_pred[:3, :3], precision=4, suppress_small=True))

    print("=" * 65 + "\n")


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":

    model = KagomePropertyNQS(
            num_sites=num_sites, 
            obs_shapes=obs_shapes, 
            hidden_dim=128, 
            head_dim=64,
        )
    
    h5_file = "three_site_Kagome_Hubbard_296.h5"

    # iterate through all kagome files
    # instantiate model in main instead of training a new model everytime


    X_data, y_psi, observables_true, param_vector, num_sites = load_kagome_dataset(h5_file)

    observables_true, pred_heads, head_losses = train_kagome_nqs(
        X_data, y_psi, observables_true, param_vector,
        num_epochs=1500, lr=1e-3, w_psi=1.0, model
    )

    display_results(observables_true, pred_heads, head_losses, num_sites)