import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import numpy as np

# TeNPy imports
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain


# ==========================================
# 1. TENPY DATA & GROUND TRUTH GENERATION
# ==========================================
def get_tenpy_ground_truth(L=10, J=1.0, g=1.0, bond_dim=32):
    print(f"--- Step 0: Solving Exact Ground State via TeNPy DMRG (L={L}) ---")

    # Define TFI Model in TeNPy (Open Boundary Conditions)
    model_params = {
        "L": L,
        "J": J,
        "g": g,
        "bc_MPS": "finite",
        "conserve": None,  # No symmetry constraints to match full Hilbert space
    }
    model = TFIChain(model_params)

    psi_tenpy = MPS.from_product_state(
        model.lat.mps_sites(), ["up"] * L, "finite"
    )

    # Run DMRG
    dmrg_params = {
        "mixer": True,
        "trunc_params": {"chi_max": bond_dim, "svd_min": 1e-10},
        "max_sweeps": 50,
    }
    results = dmrg.run(psi_tenpy, model, dmrg_params)
    exact_energy = results["E"]

    # Contract full MPS into an L-site tensor and convert to a NumPy array
    theta = psi_tenpy.get_theta(i=0, n=L)
    psi_exact_vec = theta.to_ndarray().reshape(-1)

    # Normalize vector explicitly
    psi_exact_vec /= np.linalg.norm(psi_exact_vec)

    print(f"TeNPy DMRG Ground State Energy: {exact_energy:.8f}")
    return exact_energy, psi_exact_vec


# ==========================================
# 2. MODEL CLASS DEFINITION
# ==========================================
class UnsupervisedNQS(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # Input shape: (batch_size, L) with values in {-1, 1}
        h = nn.Dense(self.hidden_dim, param_dtype=jnp.float64)(x)
        h = nn.tanh(h)
        log_psi = nn.Dense(1, param_dtype=jnp.float64)(h)
        return jnp.squeeze(log_psi, axis=-1)


# ==========================================
# 3. UNSUPERVISED TRAINING (VMC)
# ==========================================
def train_vmc_nqs(L=10, n_iter=600, lr=1e-2):
    print(f"\n--- Step 1: Setting up NetKet Hilbert Space & Hamiltonian for L={L} ---")
    
    # Define NetKet Spin Hilbert space (S = 1/2)
    hi = nk.hilbert.Spin(s=1/2, N=L)
    
    # Define Transverse Field Ising Model (TFI)
    graph = nk.graph.Chain(length=L, pbc=False)
    hamiltonian = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0, J=1.0)

    # RBM Model with param_dtype
    model = nk.models.RBM(alpha=1, param_dtype=float)
    
    # MCMC Sampler
    sampler = nk.sampler.MetropolisLocal(hi)
    
    # Variational Monte Carlo State
    vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

    # Optimizer & Driver
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=0.01)  # Stochastic Reconfiguration
    
    driver = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)

    print("\n--- Step 2: Training Unsupervised NQS via Variational Principle ---")
    
    # Run optimization for n_iter steps
    driver.run(n_iter=n_iter)

    return hi, hamiltonian, vstate


# ==========================================
# 4. BENCHMARK & FIDELITY CHECK AGAINST TENPY
# ==========================================
def verify_against_tenpy_aligned(hi, hamiltonian, vstate, exact_energy, psi_exact_vec, L):
    print("\n" + "=" * 60)
    print("      VERIFICATION: UNSUPERVISED VMC vs. TENPY GROUND TRUTH")
    print("=" * 60)

    # 1. Compare Energies
    vmc_energy = float(vstate.expect(hamiltonian).mean.real)
    energy_err = abs(vmc_energy - exact_energy)
    print(f"TeNPy DMRG Energy  : {exact_energy:.8f}")
    print(f"Learned VMC Energy : {vmc_energy:.8f}")
    print(f"Absolute Energy Err: {energy_err:.2e}")

    # 2. Get NetKet wave function on all Hilbert basis states
    all_states_nk = hi.all_states()  # Shape: (2^L, L) with values -1, +1
    log_psi = np.array(vstate.log_value(all_states_nk))
    psi_vmc = np.exp(log_psi - np.max(log_psi))
    psi_vmc /= np.linalg.norm(psi_vmc)

    # 3. Correct Spin Mapping: NetKet (-1, +1) -> TeNPy (1=down, 0=up)
    # NetKet -1 -> TeNPy 1
    # NetKet +1 -> TeNPy 0
    binary_states_tenpy = (all_states_nk == -1).astype(int)

    # 4. Check Big-Endian vs Little-Endian indexing for TeNPy vector
    pow_big = 2 ** np.arange(L)[::-1]
    idx_big = np.dot(binary_states_tenpy, pow_big)
    psi_big = np.zeros_like(psi_vmc)
    psi_big[idx_big] = psi_vmc

    pow_lit = 2 ** np.arange(L)
    idx_lit = np.dot(binary_states_tenpy, pow_lit)
    psi_lit = np.zeros_like(psi_vmc)
    psi_lit[idx_lit] = psi_vmc

    fid_big = np.abs(np.vdot(psi_big, psi_exact_vec)) ** 2
    fid_lit = np.abs(np.vdot(psi_lit, psi_exact_vec)) ** 2

    print(f"\nFidelity (Big-Endian Map)   : {fid_big:.6f}")
    print(f"Fidelity (Little-Endian Map): {fid_lit:.6f}")
    print(f"--> True Aligned State Fidelity: {max(fid_big, fid_lit):.6f}")
    print("=" * 60 + "\n")

# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    L = 10
    
    # 1. Generate ground truth with TeNPy
    exact_energy, psi_exact_vec = get_tenpy_ground_truth(L=L, J=1.0, g=1.0)

    # 2. Train unsupervised NetKet model
    hi, hamiltonian, vstate = train_vmc_nqs(L=L, n_iter=600, lr=1e-2)

    # 3. Check VMC performance against TeNPy
verify_against_tenpy_aligned(hi, hamiltonian, vstate, exact_energy, psi_exact_vec, L)