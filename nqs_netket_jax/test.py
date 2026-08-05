import netket as nk
import flax.linen as nn
import jax.numpy as jnp
import optax

# 1. Define Hilbert space and Lattice (1D Ising model with N spins)
N = 20
graph = nk.graph.Chain(length=N, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)

# 2. Define Hamiltonian (transverse-field Ising model: -J ∑ Z_i Z_{i+1} - h ∑ X_i)
# J = 1.0, h = 1.0 (critical point)
hamiltonian = nk.operator.Ising(hilbert=hilbert, graph=graph, h=1.0, J=1.0)

# 3. Define the NQS Model using Flax (JAX neural network library)
class RBM(nn.Module):
    alpha: int = 1  # Feature density ratio

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, N) with values in {-1, 1}
        # Output is log(Ψ(x))
        dense = nn.Dense(features=self.alpha * x.shape[-1], param_dtype=jnp.complex64)
        return jnp.sum(jnp.log(jnp.cosh(dense(x))), axis=-1)

model = RBM(alpha=1)

# 4. Set up the Sampler (Metropolis Local Spin Flips)
sampler = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains=16)

# 5. Set up the Optimizer and Variational State (VMC)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=1000  # Number of Monte Carlo samples per step
)

optimizer = nk.optimizer.Sgd(learning_rate=0.01)

# 6. Run Variational Monte Carlo Optimization
driver = nk.driver.VMC(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    variational_state=vstate
)

# Run for 100 epochs
driver.run(n_iter=100)