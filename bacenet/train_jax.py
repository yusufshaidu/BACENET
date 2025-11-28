import jax
import optax
from flax import nnx

class ModelTraining:
    def __init__(self, learning_rate: float):
        self._learning_rate = learning_rate
        tx = optax.adam(self._learning_rate)
        self._optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        self.fcost = 1.0
        self.efcost = 1.0

    def loss_fun(self,
        preditions: jax.Array,
        target: jax.Array):
      loss = optax.ls_loss(predictions=predictions, targets=targets)
      ).mean()
      return loss
    def total_loss_fun(self,
        preditions_e: jax.Array,
        preditions_f: jax.Array,
        target_e: jax.Array,
        target_f: jax.Array):
      loss = self.ecost * self.loss_fun(
        preditions_e, target_e)
      loss += self.fcost * self.loss_fun(
        preditions_f, target_f)
      return loss

    @nnx.jit  # JIT-compile the function
    def train_step(self, data):
        outs = self.model(data)
        #unpack preditions, data should also contain target energies and forces
        preditions_e = outs['energy']
        preditions_f = outs['forces']

        loss_gradient = nnx.grad(self.total_loss_fun, has_aux=True)  # gradient transform!
        grads = loss_gradient(preditions_e, preditions_f,
                            target_e, target_f)
        self._optimizer.update(grads)  # inplace update

## Training step

for i in range(301):  # 300 training epochs
  train_step(model, optimizer, images_train, label_train)
  if i % 50 == 0:  # Print metrics.
    loss, _ = loss_fun(model, images_test, label_test)
    print(f"epoch {i}: loss={loss:.2f}")
