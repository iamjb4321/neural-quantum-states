import numpy as np
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIModel
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import get_full_wavefunction

# 1. Setup a 1D Transverse Field Ising Model in TeNPy
L = 10  # System size
model_params = {'L': L, 'J': 1.0, 'g': 1.0, 'bc_MPS': 'finite'}
model = TFIModel(model_params)

# 2. Initialize starting state (MPS product state)
init_state = ["up"] * L
psi = MPS.from_product_state(
    model.lat.mps_sites(), 
    init_state, 
    bc=model.lat.bc_MPS
)

# 3. Run DMRG (modifies `psi` in-place)
dmrg_params = {'trunc_params': {'chi_max': 30, 'svd_min': 1e-10}, 'verbose': 0}
info = dmrg.run(psi, model, dmrg_params)

# 4. Extract dense wave function directly from the optimized `psi`
theta = get_full_wavefunction(psi)
psi_dense = theta.to_ndarray() if hasattr(theta, 'to_ndarray') else np.array(theta)

# Flatten to shape (2^L,)
y_data = psi_dense.reshape(-1).astype(np.float32)

# Generate corresponding spin configurations (-1, +1) matching TeNPy basis ordering
basis_indices = np.arange(2**L)
binary_matrix = ((basis_indices[:, None] >> np.arange(L)[::-1]) & 1)
X_data = (2 * binary_matrix - 1).astype(np.float32)  # Shape: (2^L, L)

# Normalize target wave function
y_data = y_data / np.linalg.norm(y_data)

# Save to file
np.savez_compressed('tenpy_dataset.npz', X=X_data, y=y_data)
print(f"Successfully saved dataset with shape X={X_data.shape}, y={y_data.shape} to 'tenpy_dataset.npz'")
