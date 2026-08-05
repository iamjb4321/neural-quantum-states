import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import exact_diag


def get_tenpy_ground_truth(L=10, J=1.0, g=1.0, bond_dim=32):
    print(f"--- Step 0: Solving Exact Ground State via TeNPy DMRG (L={L}) ---")

    model_params = {
        "L": L,
        "J": J,
        "g": g,
        "bc_MPS": "finite",
        "conserve": None,
    }
    model = TFIChain(model_params)

    psi_tenpy = MPS.from_product_state(
        model.lat.mps_sites(), ["up"] * L, "finite"
    )

    dmrg_params = {
        "mixer": True,
        "trunc_params": {"chi_max": bond_dim, "svd_min": 1e-10},
        "max_sweeps": 50,
    }
    results = dmrg.run(psi_tenpy, model, dmrg_params)
    exact_energy = results["E"]

    # Extract state vector using TeNPy's exact_diag helper
    theta = exact_diag.get_full_wavefunction(psi_tenpy)

    # Safely handle both npc.Array and standard numpy.ndarray
    if hasattr(theta, "to_ndarray"):
        psi_exact_vec = theta.to_ndarray().reshape(-1)
    else:
        psi_exact_vec = np.asarray(theta).reshape(-1)

    psi_exact_vec /= np.linalg.norm(psi_exact_vec)

    print(f"TeNPy DMRG Ground State Energy: {exact_energy:.8f}")
    return exact_energy, psi_exact_vec


def train_vmc_nqs(L=10, n_iter=600, lr=1e-2):
    print(f"\n--- Step 1: Setting up NetKet Hilbert Space & Hamiltonian for L={L} ---")
    
    hi = nk.hilbert.Spin(s=1/2, N=L)
    graph = nk.graph.Chain(length=L, pbc=False)
    hamiltonian = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0, J=1.0)

    model = nk.models.RBM(alpha=1, param_dtype=float)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=0.01)
    
    driver = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=sr)

    print("\n--- Step 2: Training Unsupervised NQS via Variational Principle ---")
    driver.run(n_iter=n_iter)

    return hi, hamiltonian, vstate


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

    # 2. Extract full wave function vector from NetKet
    all_states_nk = np.array(hi.all_states())  # (2^L, L) containing +1, -1
    log_psi = np.array(vstate.log_value(all_states_nk))
    psi_vmc = np.exp(log_psi - np.max(log_psi))
    psi_vmc /= np.linalg.norm(psi_vmc)

    # --- FIX 2: Map NetKet spin states (+1, -1) to TeNPy site indices (0, 1) ---
    # NetKet +1 (up) -> 0, -1 (down) -> 1
    tenpy_indices = ((1 - all_states_nk) // 2).astype(int)

    # Big-endian decimal indexing matching TeNPy's flattened order
    powers_of_two = 2 ** np.arange(L)[::-1]
    flat_indices = np.dot(tenpy_indices, powers_of_two)

    # Re-index NetKet vector into standard TeNPy basis ordering
    psi_vmc_aligned = np.zeros_like(psi_vmc)
    psi_vmc_aligned[flat_indices] = psi_vmc

    # 3. Calculate Fidelity Overlap
    overlap = np.abs(np.vdot(psi_vmc_aligned, psi_exact_vec))
    fidelity = overlap**2

    print(f"\nState Fidelity Overlap |<Ψ_VMC | Ψ_TeNPy>|²: {fidelity:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    L = 10
    exact_energy, psi_exact_vec = get_tenpy_ground_truth(L=L, J=1.0, g=1.0)
    hi, hamiltonian, vstate = train_vmc_nqs(L=L, n_iter=600, lr=1e-2)
    verify_against_tenpy_aligned(hi, hamiltonian, vstate, exact_energy, psi_exact_vec, L)
    