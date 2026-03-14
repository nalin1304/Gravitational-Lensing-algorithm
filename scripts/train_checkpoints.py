"""Train PINN and LensFinder models on synthetic NFW data and save checkpoints."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===== 1. Train PINN (JAX/Equinox) =====
print("=== Training PINN on synthetic NFW convergence maps ===")
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from src.ml.pinn import PhysicsInformedNN
from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem

np.random.seed(42)
n_samples = 200
grid_size = 64

kappa_maps = []
true_params = []

for i in range(n_samples):
    M_vir = 10 ** (np.random.uniform(11.5, 13.5))
    c = np.random.uniform(3, 15)
    z_l = np.random.uniform(0.1, 0.5)
    z_s = z_l + np.random.uniform(0.3, 1.5)

    ls = LensSystem(z_lens=z_l, z_source=z_s)
    nfw = NFWProfile(M_vir=M_vir, concentration=c, lens_system=ls)
    theta = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(theta, theta)
    r = np.sqrt(X ** 2 + Y ** 2) + 1e-10
    kappa = nfw.convergence(X, Y)
    kappa = np.clip(kappa, 0, 10)

    kappa_maps.append(kappa.reshape(1, grid_size, grid_size))
    true_params.append([
        np.log10(M_vir) / 14.0,
        c / 15.0,
        0.0,
        0.0,
        0.5,
    ])

X_train = jnp.array(np.array(kappa_maps), dtype=jnp.float32)
Y_train = jnp.array(np.array(true_params), dtype=jnp.float32)
print(f"  Training data: {X_train.shape} maps, {Y_train.shape} params")

key = jax.random.PRNGKey(42)
model = PhysicsInformedNN(key=key)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def loss_fn(model, x, y):
    def predict_one(xi):
        params, _class_logits = model(xi)
        return params
    pred = jax.vmap(predict_one)(x)
    return jnp.mean((pred - y) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state_new = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state_new, loss


batch_size = 32
n_epochs = 100
t0 = time.time()

for epoch in range(n_epochs):
    perm = np.random.permutation(n_samples)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, n_samples, batch_size):
        idx = perm[start : start + batch_size]
        xb = X_train[idx]
        yb = Y_train[idx]
        model, opt_state, batch_loss = train_step(model, opt_state, xb, yb)
        epoch_loss += float(batch_loss)
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        avg = epoch_loss / n_batches
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.6f}")

elapsed = time.time() - t0
print(f"  PINN training completed in {elapsed:.1f}s")

os.makedirs("models", exist_ok=True)
eqx.tree_serialise_leaves("models/pinn_best.eqx", model)
print("  Saved models/pinn_best.eqx")

# Verify
key2 = jax.random.PRNGKey(0)
model_test = PhysicsInformedNN(key=key2)
model_test = eqx.tree_deserialise_leaves("models/pinn_best.eqx", model_test)
test_pred, test_cls = model_test(X_train[0])
print(f"  Verification: pred shape={test_pred.shape}, class shape={test_cls.shape}")
print(f"  Sample prediction: {test_pred}")
print()


# ===== 2. Train LensFinder (PyTorch) =====
print("=== Training LensFinder on synthetic lens candidates ===")
import torch
import torch.nn as nn
from src.ml.lens_finder import LensNetModel

torch.manual_seed(42)
np.random.seed(42)
n_lens = 150
n_nonlens = 150
n_total = n_lens + n_nonlens

images = []
labels = []

for i in range(n_lens):
    M_vir = 10 ** (np.random.uniform(11.5, 13.5))
    c = np.random.uniform(3, 15)
    z_l = np.random.uniform(0.1, 0.5)
    z_s = z_l + np.random.uniform(0.3, 1.5)
    ls = LensSystem(z_lens=z_l, z_source=z_s)
    nfw = NFWProfile(M_vir=M_vir, concentration=c, lens_system=ls)
    theta = np.linspace(-3, 3, 64)
    X, Y = np.meshgrid(theta, theta)
    r = np.sqrt(X ** 2 + Y ** 2) + 1e-10
    kappa = nfw.convergence(X, Y)
    kappa = np.clip(kappa, 0, 10) + np.random.normal(0, 0.02, kappa.shape)
    kappa = np.clip(kappa, 0, None)
    images.append(kappa)
    cx = np.clip(0.5 + np.random.normal(0, 0.05), 0.1, 0.9)
    cy = np.clip(0.5 + np.random.normal(0, 0.05), 0.1, 0.9)
    w, h = np.random.uniform(0.2, 0.5), np.random.uniform(0.2, 0.5)
    labels.append([1.0, cx, cy, w, h])

for i in range(n_nonlens):
    noise = np.random.exponential(0.05, (64, 64))
    images.append(noise)
    labels.append([0.0, 0.5, 0.5, 0.1, 0.1])

images_t = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1)
labels_t = torch.tensor(np.array(labels), dtype=torch.float32)

for i in range(len(images_t)):
    mx = images_t[i].max()
    if mx > 0:
        images_t[i] = images_t[i] / mx

print(f"  Training data: {images_t.shape} images, {labels_t.shape} labels")

finder_model = LensNetModel()
optimizer_pt = torch.optim.Adam(finder_model.parameters(), lr=1e-3)
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

n_epochs_pt = 80
t0 = time.time()

for epoch in range(n_epochs_pt):
    perm = torch.randperm(n_total)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, n_total, 32):
        idx = perm[start : start + 32]
        xb = images_t[idx]
        yb = labels_t[idx]

        pred = finder_model(xb)
        obj_loss = bce_loss(pred[:, 0], yb[:, 0])
        mask = yb[:, 0].unsqueeze(1)
        bbox_loss = mse_loss(pred[:, 1:] * mask, yb[:, 1:] * mask)
        loss = obj_loss + 5.0 * bbox_loss

        optimizer_pt.zero_grad()
        loss.backward()
        optimizer_pt.step()
        epoch_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs_pt}  loss={epoch_loss / n_batches:.6f}")

elapsed = time.time() - t0
print(f"  LensFinder training completed in {elapsed:.1f}s")

torch.save(finder_model.state_dict(), "models/lens_finder.pt")
print("  Saved models/lens_finder.pt")

# Verify
finder_test = LensNetModel()
finder_test.load_state_dict(torch.load("models/lens_finder.pt", map_location="cpu", weights_only=True))
finder_test.eval()
with torch.no_grad():
    test_out = finder_test(images_t[:1])
print(f"  Verification: output shape={test_out.shape}, values={test_out[0].numpy()}")
print()
print("=== Both models trained and saved successfully ===")
