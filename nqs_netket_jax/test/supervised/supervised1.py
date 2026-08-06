import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# TeNPy imports
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIModel
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import get_full_wavefunction


# ==========================================
# 1. DATA GENERATION STEP (TeNPy DMRG)
# ==========================================
def generate_data(L=10, filename='tenpy_dataset.npz'):
    print(f"--- Step 1: Running DMRG in TeNPy for L={L} ---")
    
    model_params = {'L': L, 'J': 1.0, 'g': 1.0, 'bc_MPS': 'finite'}
    model = TFIModel(model_params)

    init_state = ["up"] * L
    psi = MPS.from_product_state(model.lat.mps_sites(), init_state, bc=model.lat.bc_MPS)

    psi_copy = psi.copy()

    dmrg_params = {'trunc_params': {'chi_max': 30, 'svd_min': 1e-10}, 'verbose': 0}
    dmrg.run(psi, model, dmrg_params)

    theta = get_full_wavefunction(psi)
    psi_dense = theta.to_ndarray() if hasattr(theta, 'to_ndarray') else np.array(theta)
    y_data = psi_dense.reshape(-1).astype(np.float32)

    basis_indices = np.arange(2**L)
    binary_matrix = ((basis_indices[:, None] >> np.arange(L)[::-1]) & 1)
    X_data = (2 * binary_matrix - 1).astype(np.float32)

    y_data = y_data / np.linalg.norm(y_data)

    np.savez_compressed(filename, X=X_data, y=y_data)
    print(f"Dataset successfully saved to '{filename}' | Shapes: X={X_data.shape}, y={y_data.shape}\n")


# ==========================================
# 2. MODEL CLASS DEFINITION
# ==========================================
class SupervisedNQS(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.tanh(h)
        log_psi = nn.Dense(1)(h)
        return jnp.squeeze(log_psi, axis=-1)


# ==========================================
# 3. SUPERVISED TRAINING (JAX)
# ==========================================
def train_nqs(filename='tenpy_dataset.npz', num_epochs=1000, lr=1e-3):
    print("--- Step 2: Training Supervised NQS in JAX ---")
    
    # Load dataset
    data = np.load(filename)
    X_data = jnp.array(data['X'])
    y_data = jnp.array(data['y'])
    L = X_data.shape[1]

    # Instantiate model inside function scope
    model = SupervisedNQS(hidden_dim=64)
    optimizer = optax.adam(learning_rate=lr)

    # Define loss function inside train_nqs so it has access to `model`
    @jax.jit
    def loss_fn(params, x_batch, y_batch):
        log_psi = model.apply({'params': params}, x_batch)
        
        # Shift max for exponentiation stability
        log_psi_shifted = log_psi - jnp.max(log_psi)
        psi = jnp.exp(log_psi_shifted)
        
        overlap = jnp.dot(psi, y_batch)
        norm_model = jnp.linalg.norm(psi)
        norm_true = jnp.linalg.norm(y_batch)
        
        fidelity = (overlap**2) / (norm_model**2 * norm_true**2 + 1e-12)
        return 1.0 - fidelity

    @jax.jit
    def train_step(params, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, X_data, y_data)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    # Initialize PRNG and parameters
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, X_data[:10])['params']
    opt_state = optimizer.init(params)

    # Optimization Loop
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step(params, opt_state)
        
        if (epoch + 1) % 100 == 0:
            fidelity = 1.0 - float(loss)
            print(f"Epoch {epoch+1:4d} | Overlap Loss: {loss:.6e} | Fidelity: {fidelity:.6f}")

    print("\nTraining complete!")

    # Extract learned normalized wavefunction vector
    log_psi_pred = model.apply({'params': params}, X_data)
    log_psi_shifted = log_psi_pred - jnp.max(log_psi_pred)
    psi_nqs = jnp.exp(log_psi_shifted)
    psi_nqs = psi_nqs / jnp.linalg.norm(psi_nqs)

    return y_data, psi_nqs, X_data, L


# ==========================================
# 4. MEASUREMENT & PROPERTY EXTRACTION HELPERS
# ==========================================
def measure_local_sz(psi, X_data):
    probs = jnp.abs(psi) ** 2
    return jnp.dot(probs, X_data)

def measure_local_sx(psi, X_data, L):
    sx_expectations = []
    for i in range(L):
        flip_mask = 1 << (L - 1 - i)
        basis_indices = jnp.arange(2**L)
        flipped_indices = basis_indices ^ flip_mask
        
        psi_flipped = psi[flipped_indices]
        sx_i = jnp.dot(psi, psi_flipped)
        sx_expectations.append(float(sx_i))
    return jnp.array(sx_expectations)

def measure_sz_sz_correlations(psi, X_data, L):
    probs = jnp.abs(psi) ** 2
    corr_matrix = jnp.zeros((L, L))
    for i in range(L):
        for j in range(L):
            spin_prod = X_data[:, i] * X_data[:, j]
            corr_val = jnp.dot(probs, spin_prod)
            corr_matrix = corr_matrix.at[i, j].set(corr_val)
    return corr_matrix


# ==========================================
# 5. PROPERTY COMPARISON
# ==========================================
def compare_properties(y_true, psi_nqs, X_data, L):
    print("\n" + "="*50)
    print("      PHYSICAL OBSERVABLES: DMRG vs NQS")
    print("="*50)
    
    # 1. Local Magnetization <σ^z_i>
    sz_true = measure_local_sz(y_true, X_data)
    sz_nqs = measure_local_sz(psi_nqs, X_data)
    
    print("\n[1] Local Magnetization <σ^z_i> across chain:")
    print(f"  Site i  | True DMRG | Learned NQS | Abs Error")
    print("  -------------------------------------------")
    for i in range(L):
        err = abs(sz_true[i] - sz_nqs[i])
        print(f"    {i:2d}    |  {sz_true[i]:+.5f}  |   {sz_nqs[i]:+.5f}   |  {err:.2e}")
        
    # 2. Transverse Magnetization <σ^x_i>
    sx_true = measure_local_sx(y_true, X_data, L)
    sx_nqs = measure_local_sx(psi_nqs, X_data, L)
    
    print("\n[2] Transverse Magnetization <σ^x_i> across chain:")
    print(f"  Site i  | True DMRG | Learned NQS | Abs Error")
    print("  -------------------------------------------")
    for i in range(L):
        err = abs(sx_true[i] - sx_nqs[i])
        print(f"    {i:2d}    |  {sx_true[i]:+.5f}  |   {sx_nqs[i]:+.5f}   |  {err:.2e}")

    # 3. Nearest-Neighbor Spin Correlation <σ^z_i σ^z_{i+1}>
    corr_true = measure_sz_sz_correlations(y_true, X_data, L)
    corr_nqs = measure_sz_sz_correlations(psi_nqs, X_data, L)
    
    print("\n[3] Nearest-Neighbor Correlations <σ^z_i σ^z_{i+1}>:")
    print(f"  Pair (i, i+1) | True DMRG | Learned NQS | Abs Error")
    print("  --------------------------------------------------")
    for i in range(L - 1):
        c_true = corr_true[i, i+1]
        c_nqs = corr_nqs[i, i+1]
        print(f"    ({i:2d}, {i+1:2d})     |  {c_true:+.5f}  |   {c_nqs:+.5f}   |  {abs(c_true - c_nqs):.2e}")
    print("="*50 + "\n")


# ==========================================
# 6. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    
    dataset_file = 'tenpy_dataset.npz'
    
    generate_data(L=10, filename=dataset_file)
    y_true, psi_nqs, X_data, L = train_nqs(filename=dataset_file, num_epochs=1000)
    compare_properties(y_true, psi_nqs, X_data, L)
    '''
    
    filename='tenpy_dataset.npz'
    num_epochs=1000
    lr=1e-3

    print("--- Step 2: Training Supervised NQS in JAX ---")
        
        # Load dataset
    data = np.load(filename)
    X_data = jnp.array(data['X'])
    y_data = jnp.array(data['y'])
    L = X_data.shape[1]
    print(f"X_data: {X_data.shape}")
    print(f"Y_data: {y_data.shape}")
    '''