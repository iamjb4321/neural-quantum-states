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
        
        # Extract scalar Hamiltonian parameters for parameter-based inputs
        U = float(p["U"][()]) if "U" in p else 0.0
        t1 = float(p["t1"][()]) if "t1" in p else 1.0
        t2 = float(p["t2"][()]) if "t2" in p else 0.0
        theta_x = float(p["theta_x"][()]) if "theta_x" in p else 0.0
        theta_y = float(p["theta_y"][()]) if "theta_y" in p else 0.0
        
        param_vector = np.array([U, t1, t2, theta_x, theta_y], dtype=np.float32)

        # --- B. Read Observables ---
        obs = f["observables"]
        
        # 1. Charge Density / Local Occupancy
        if "charge_density" in obs:
            density_true = np.array(obs["charge_density"], dtype=np.float32)
        else:
            raise KeyError("Neither 'charge_density' nor 'S_q_charge' found in observables.")

        if "S_q_charge" in obs:
            s_q_charge_true = np.array(obs["S_q_charge"], dtype=np.float32)
        else:
            raise KeyError("Neither 'charge_density' nor 'S_q_charge' found in observables.")

        if "S_q_spin" in obs:
            S_q_spin_true = np.array(obs["S_q_spin"], dtype=np.float32)
        else:
            raise KeyError("Neither 'charge_density' nor 'S_q_spin' found in observables.")


        # Ensure 1D array for local sites
        density_true = np.squeeze(density_true)
        num_sites = len(density_true)

        # 2. Spin Correlations
        if "spin_corr" in obs:
            spin_corr_true = np.array(obs["spin_corr"], dtype=np.float32)
        else:
            raise KeyError("No 'spin_corr' found in observables.")
        

        if "charge_corr" in obs:
            charge_corr_true = np.array(obs["charge_corr"], dtype=np.float32)
        else:
            raise KeyError("No 'charge_corr' found in observables.")

        # Squeeze down to 2D matrix (num_sites, num_sites)
        #corr_true = np.squeeze(corr_true)

        if "double_occupancy" in obs:
            double_occupancy_true = np.array(obs["double_occupancy"], dtype=np.float32)
        else:
            raise KeyError("No 'double_occupancy' found in observables.")

        observables = {
            'charge_density': density_true,
            'double_occupancy': double_occupancy_true,
            's_q_charge': s_q_charge_true,
            'S_q_spin': S_q_spin_true,
            'spin_corr': spin_corr_true,
            'charge_corr':charge_corr_true
        }


        # --- C. Read or Deserialise Wavefunction ---
        y_psi = None
        if "psi" in f and HAS_TENPY:
            print("Deserializing native TeNPy MPS object from HDF5...")
            psi_mps = h5_tenpy.load_from_hdf5(f, "psi")
            psi_dense = get_full_wavefunction(psi_mps).to_ndarray()
            y_psi = psi_dense.reshape(-1).astype(np.float32)

    # --- D. Build Basis Matrix X if State Vector exists ---
    if y_psi is not None:
        dim = len(y_psi)
        y_psi = y_psi / np.linalg.norm(y_psi)
        
        # Construct binary configuration matrix for basis dimension
        basis_indices = np.arange(dim)
        binary_matrix = (basis_indices[:, None] >> np.arange(num_sites)[::-1]) & 1
        X_data = (2 * binary_matrix - 1).astype(np.float32)
    else:
        print("Note: No explicit state vector found in file. Training on observables directly.")
        X_data = None

    print(f"Loaded successfully!")
    print(f"Lattice Sites: {num_sites} | Params Vector: {param_vector}")
    print(f"Shapes -> Density: {density_true.shape}, Spin Corr: {corr_true.shape}\n")

    #return X_data, y_psi, density_true, s_q_charge_true, corr_true, param_vector, num_sites
    return X_data, y_psi, observables, param_vector, num_sites



# ==========================================
# 2. MULTI-TASK FLAX MODEL
# ==========================================
class KagomePropertyNQS(nn.Module):
    num_sites: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x_input):
        # x_input can be either configuration batch (batch_size, num_sites) 
        # or Hamiltonian parameter vector (1, num_params)
        h = nn.Dense(self.hidden_dim)(x_input)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.tanh(h)

        # Head 1: Wavefunction Amplitudes log|psi(sigma)| -> (Batch, 1) -> (Batch,)
        log_psi = jnp.squeeze(nn.Dense(1)(h), axis=-1)

        # Aggregated representation for global property prediction
        h_global = jnp.mean(h, axis=0, keepdims=True)

        # Head 2: Direct Local Charge Density prediction -> (num_sites,)
        density_pred = jnp.squeeze(nn.Dense(self.num_sites)(h_global), axis=0)

        # Head 3: Direct Spin Correlation Matrix prediction -> (num_sites, num_sites)
        corr_flat = nn.Dense(self.num_sites * self.num_sites)(h_global)
        corr_pred = jnp.reshape(corr_flat, (self.num_sites, self.num_sites))

        return log_psi, density_pred, corr_pred


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_kagome_nqs(
    X_data, y_psi, density_true, corr_true, param_vector, num_sites,
    num_epochs=1200, lr=1e-3, weights=(1.0, 0.5, 0.5)
):
    print("--- Training Multi-Task Neural Network in JAX ---")

    w_psi, w_dens, w_corr = weights
    model = KagomePropertyNQS(num_sites=num_sites, hidden_dim=128)
    optimizer = optax.adam(learning_rate=lr)

    # Convert targets to JAX arrays
    target_density = jnp.array(density_true)
    target_corr = jnp.array(corr_true)

    has_psi = X_data is not None and y_psi is not None
    if has_psi:
        X_inputs = jnp.array(X_data)
        y_targets = jnp.array(y_psi)
    else:
        # Fallback input: repeat the Hamiltonian parameter vector across dummy batch size
        X_inputs = jnp.repeat(jnp.array(param_vector)[None, :], 16, axis=0)
        y_targets = None

    @jax.jit
    def loss_fn(params, x_batch, y_batch):
        log_psi, density_pred, corr_pred = model.apply({"params": params}, x_batch)

        # 1. Wavefunction Overlap Loss (if state vector is present)
        if has_psi:
            log_psi_shifted = log_psi - jnp.max(log_psi)
            psi = jnp.exp(log_psi_shifted)
            overlap = jnp.dot(psi, y_batch)
            norm_model = jnp.linalg.norm(psi)
            norm_true = jnp.linalg.norm(y_batch)
            loss_psi = 1.0 - ((overlap**2) / (norm_model**2 * norm_true**2 + 1e-12))
        else:
            loss_psi = 0.0

        # 2. MSE Loss on Charge Density
        loss_dens = jnp.mean((density_pred - target_density) ** 2)

        # 3. MSE Loss on Spin Correlation Matrix
        loss_corr = jnp.mean((corr_pred - target_corr) ** 2)

        total_loss = w_psi * loss_psi + w_dens * loss_dens + w_corr * loss_corr
        return total_loss, (loss_psi, loss_dens, loss_corr, density_pred, corr_pred)

    @jax.jit
    def train_step(params, opt_state):
        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, X_inputs, y_targets
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, total_loss, aux

    # Initialize Parameters
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, X_inputs[:10])["params"]
    opt_state = optimizer.init(params)

    # Optimization Loop
    for epoch in range(num_epochs):
        params, opt_state, total_loss, aux = train_step(params, opt_state)
        loss_psi, loss_dens, loss_corr, density_pred, corr_pred = aux

        if (epoch + 1) % 150 == 0:
            print(
                f"Epoch {epoch+1:4d} | Total Loss: {total_loss:.6f} | "
                f"Density MSE: {loss_dens:.2e} | Spin Corr MSE: {loss_corr:.2e}"
            )

    print("\nTraining completed successfully!")
    return density_true, density_pred, corr_true, corr_pred, num_sites


# ==========================================
# 4. PRINT COMPARISON
# ==========================================
def display_results(density_true, density_pred, corr_true, corr_pred, num_sites):
    print("\n" + "=" * 60)
    print("      KAGOME PHYSICAL PROPERTIES: TE NPY H5 vs NQS PREDICTION")
    print("=" * 60)

    # 1. Local Charge Density
    print("\n[1] Local Charge Density Across Sites:")
    print(f"  Site i  | H5 True Value | NQS Head Pred | Abs Error")
    print("  ----------------------------------------------------")
    for i in range(num_sites):
        err = abs(density_true[i] - density_pred[i])
        print(
            f"    {i:2d}    |  {density_true[i]:+.5f}    |  {density_pred[i]:+.5f}     |  {err:.2e}"
        )

    # 2. Spin Correlation Matrix Preview
    print("\n[2] Spin Correlation Matrix Comparison (First 3x3 Block):")
    print("\n--- H5 True Spin Correlations ---")
    print(np.array2string(np.array(corr_true[:3, :3]), precision=4, suppress_small=True))
    
    print("\n--- NQS Predicted Spin Correlations ---")
    print(np.array2string(np.array(corr_pred[:3, :3]), precision=4, suppress_small=True))
    print("=" * 60 + "\n")


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Update path if necessary
    h5_file = "data_outputs/wavefunctions/three_site_kagome_001.h5"

    # Step 1: Load HDF5 file
    X_data, y_psi, density_true, corr_true, param_vector, num_sites = load_kagome_dataset(h5_file)

    # Step 2: Run training
    density_true, density_pred, corr_true, corr_pred, num_sites = train_kagome_nqs(
        X_data, y_psi, density_true, corr_true, param_vector, num_sites,
        num_epochs=1200, lr=1e-3, weights=(1.0, 0.5, 0.5)
    )

    # Step 3: Print comparison metrics
    display_results(density_true, density_pred, corr_true, corr_pred, num_sites)