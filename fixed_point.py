# Differentiable iterative fixed point for a residual layer
# with Anderson's acceleration.

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from jax import random
import time


_LOOKBACK = 5
_VECTOR_DIM = 1000


@jit
def anderson_update(G_lookback, X_lookback, z_new):
    LR_X = G_lookback[1:_LOOKBACK] - G_lookback[0]
    A = jnp.einsum('md, nd-> mn', LR_X, LR_X)
    B = jnp.einsum('ld, d -> l', LR_X, G_lookback[0])
    gamma = jax.scipy.linalg.solve(A, B)
    matrix = (X_lookback[1:_LOOKBACK] - X_lookback[0] + LR_X)
    z_new = z_new + jnp.einsum('ld,l->d', matrix, gamma)
    return z_new


class AndersonFixedPoint:

    def __init__(self, tol, max_iter, features):
        self.tol = tol
        key = random.PRNGKey(0)
        # N(0, 0.01) - important for contractive mapping.
        # sd should be set per vector dimension.
        self.projection = random.normal(
            key, shape=(features, features))/100
        self.max_iter = max_iter
        self.iterations = 0

    def linear(self, z, x):
        return self.projection@z + x

    def step(self, z, x):
        return jnp.tanh(self.linear(z, x))

    def compute_fp(self, x):
        z = jnp.tanh(x)
        X_lookback = jnp.zeros(shape=(_LOOKBACK, z.shape[0]))
        G_lookback = jnp.zeros(shape=(_LOOKBACK, z.shape[0]))
        start = time.time()
        for iter in range(self.max_iter):
            self.iterations += 1
            z_new = self.step(z, x)
            if jnp.linalg.norm(z - z_new) < self.tol:
                break
            X_lookback = jnp.roll(X_lookback, shift=1, axis=0)
            X_lookback = X_lookback.at[0].set(z)
            G_lookback = jnp.roll(G_lookback, shift=1, axis=0)
            G_lookback = G_lookback.at[0].set(z_new - z)
            if self.iterations >= _LOOKBACK:
                z_new = anderson_update(
                    G_lookback, X_lookback, z_new)
            z = z_new
        end = time.time()
        print(f'FP Iters = {self.iterations} at '
              f'{(end - start)/self.iterations} sec/iter')
        return z


fp_solver = AndersonFixedPoint(1e-5, 100, _VECTOR_DIM)
key = random.PRNGKey(100)
x = random.normal(key, shape=(_VECTOR_DIM,))
print(f'Device = {x.device()}')
fixed_point = fp_solver.compute_fp(x)
jacobian = jax.jacfwd(fp_solver.compute_fp)
print('tol = ', jnp.linalg.norm(fixed_point - fp_solver.step(fixed_point, x)))
