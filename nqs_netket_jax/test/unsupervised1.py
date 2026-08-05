import netket as nk
import jax.numpy as jnp
import flax.linen as nn
import optax


# ==========================================
# 1. MODEL CLASS DEFINITION
# ==========================================
class UnsupervisedNQS(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x input shape: (batch_size, L) with values in {-1, 1}
        h = nn.Dense(self.hidden_dim, param_dtype=jnp.float64)(x)
        h = nn.tanh(h)
        log_psi = nn.Dense(1, param_dtype=jnp.float64)(h)
        return jnp.squeeze(log_psi, axis=-1)


# ==========================================
# 2. UNSUPERVISED TRAINING (VMC)
# ==========================================
def train_vmc_nqs(L=10, n_iter=600, lr=1e-2):
    print(f"--- Step 1: Setting up Hilbert Space & Hamiltonian for L={L} ---")

    # Define Hilbert Space for spin-1/2 chain
    hi = nk.hilbert.Spin(s=0.5, N=L)

    # Transverse-Field Ising Model: H = -J * sum(Z_i Z_{i+1}) - g * sum(X_i)
    graph = nk.graph.Chain(length=L, pbc=False)  # Open boundary conditions
    hamiltonian = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0, J=1.0)

    # Instantiate model and MCMC sampler
    model = UnsupervisedNQS(hidden_dim=64)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)

    # Initialize Variational MC State (samples 1008 configurations per iteration)
    vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

    # Optimizer with Natural Gradient (Stochastic Reconfiguration)
    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=0.01)

    # Define the VMC Optimization Driver
    driver = nk.driver.VMC(
        hamiltonian, optimizer, variational_state=vstate, preconditioner=sr
    )

    print("\n--- Step 2: Training Unsupervised NQS via Variational Principle ---")
    # driver.iter yields step indices as it optimizes
    for step in driver.iter(n_iter):
        if step % 50 == 0 or step == 1:
            energy = vstate.expect(hamiltonian)
            print(
                f"Iteration {step:4d} | Energy: {energy.mean.real:.6f} ± {energy.error:.6f} | Variance: {energy.variance:.6f}"
            )

    print("\nTraining complete!")
    return hi, vstate


# ==========================================
# 3. PHYSICAL OBSERVABLE EXTRACTION
# ==========================================
def compute_physical_observables(hi, vstate, L):
    print("\n" + "=" * 50)
    print("      PHYSICAL OBSERVABLES: LEARNED VMC STATE")
    print("=" * 50)

    # 1. Local Magnetization <σ^z_i>
    print("\n[1] Local Magnetization <σ^z_i> across chain:")
    print("  Site i  |  Learned VMC Value  |  Std Error")
    print("  -------------------------------------------")
    for i in range(L):
        sz_op = nk.operator.spin.sigmaz(hi, i)
        res = vstate.expect(sz_op)
        print(f"    {i:2d}    |     {res.mean.real:+.5f}     |  {res.error:.2e}")

    # 2. Transverse Magnetization <σ^x_i>
    print("\n[2] Transverse Magnetization <σ^x_i> across chain:")
    print("  Site i  |  Learned VMC Value  |  Std Error")
    print("  -------------------------------------------")
    for i in range(L):
        sx_op = nk.operator.spin.sigmax(hi, i)
        res = vstate.expect(sx_op)
        print(f"    {i:2d}    |     {res.mean.real:+.5f}     |  {res.error:.2e}")

    # 3. Nearest-Neighbor Spin Correlation <σ^z_i σ^z_{i+1}>
    print("\n[3] Nearest-Neighbor Correlations <σ^z_i σ^z_{i+1}>:")
    print("  Pair (i, i+1) | Learned VMC Value | Std Error")
    print("  --------------------------------------------------")
    for i in range(L - 1):
        szsz_op = nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(
            hi, i + 1
        )
        res = vstate.expect(szsz_op)
        print(f"    ({i:2d}, {i+1:2d})     |     {res.mean.real:+.5f}     |  {res.error:.2e}")

    print("=" * 50 + "\n")


# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    L = 10
    hi, vstate = train_vmc_nqs(L=L, n_iter=600, lr=1e-2)
    compute_physical_observables(hi, vstate, L)
