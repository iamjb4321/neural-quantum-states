import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

# 1. Load Data
data = np.load('tenpy_dataset.npz')
X_data = jnp.array(data['X'])
y_data = jnp.array(data['y'])

L = X_data.shape[1]

# 2. Define Model
class SupervisedNQS(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.tanh(h)
        log_psi = nn.Dense(1)(h)
        return jnp.squeeze(log_psi, axis=-1)

model = SupervisedNQS(hidden_dim=64)

# 3. Corrected Loss Function: Maximize Fidelity directly!
@jax.jit
def loss_fn(params, x_batch, y_batch):
    log_psi = model.apply({'params': params}, x_batch)
    
    # Work in log-space for numerical stability before exp
    # Normalize log_psi by subtracting max log_psi
    log_psi_shifted = log_psi - jnp.max(log_psi)
    psi = jnp.exp(log_psi_shifted)
    
    # Compute overlap <Psi_model | Psi_true>
    overlap = jnp.dot(psi, y_batch)
    
    # Norms
    norm_model = jnp.linalg.norm(psi)
    norm_true = jnp.linalg.norm(y_batch)
    
    # Fidelity F = |<psi|y>|^2 / (|psi|^2 * |y|^2)
    fidelity = (overlap**2) / (norm_model**2 * norm_true**2 + 1e-12)
    
    # Minimize 1 - Fidelity (or -log(Fidelity))
    return 1.0 - fidelity

# 4. Training Step & Optimizer
optimizer = optax.adam(learning_rate=1e-3)

@jax.jit
def train_step(params, opt_state, x_batch, y_batch):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val

# 5. Initialization
rng = jax.random.PRNGKey(42)
params = model.init(rng, X_data[:10])['params']
opt_state = optimizer.init(params)

# 6. Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    params, opt_state, loss = train_step(params, opt_state, X_data, y_data)
    
    if (epoch + 1) % 100 == 0:
        fidelity = 1.0 - loss
        print(f"Epoch {epoch+1:4d} | Overlap Loss: {loss:.6e} | Fidelity: {fidelity:.6f}")