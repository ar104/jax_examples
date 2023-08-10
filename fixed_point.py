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
_MAX_ITER = 1000
_TOL = 1e-5


@jit
def anderson_update(G_lookback, X_lookback, z_new):
    LR_X = G_lookback[1:_LOOKBACK] - G_lookback[0]
    A = jnp.einsum('md, nd-> mn', LR_X, LR_X)
    B = jnp.einsum('ld, d -> l', LR_X, G_lookback[0])
    gamma = jax.scipy.linalg.solve(A, B)
    matrix = (X_lookback[1:_LOOKBACK] - X_lookback[0] + LR_X)
    z_new = z_new + jnp.einsum('ld,l->d', matrix, gamma)
    return z_new


key = random.PRNGKey(0)
# N(0, 0.01) - important for contractive mapping.
# sd should be set per vector dimension.
projection = random.normal(key, shape=(_VECTOR_DIM, _VECTOR_DIM))/100


def linear(z, x):
    return projection@z + x


def step(z, x):
    return jnp.tanh(linear(z, x))


def compute_fp(x):
    z = jnp.tanh(x)
    X_lookback = jnp.zeros(shape=(_LOOKBACK, z.shape[0]))
    G_lookback = jnp.zeros(shape=(_LOOKBACK, z.shape[0]))
    start = time.time()
    for iter in range(1, 1 + _MAX_ITER):
        z_new = step(z, x)
        if jnp.linalg.norm(z - z_new) < _TOL:
            break
        X_lookback = jnp.roll(X_lookback, shift=1, axis=0)
        X_lookback = X_lookback.at[0].set(z)
        G_lookback = jnp.roll(G_lookback, shift=1, axis=0)
        G_lookback = G_lookback.at[0].set(z_new - z)
        if iter >= _LOOKBACK:
            z_new = anderson_update(
                G_lookback, X_lookback, z_new)
        z = z_new
    end = time.time()
    print(f'FP Iters = {iter} at '
          f'{(end - start)/iter} sec/iter')
    return z


def compute_gradient(z, x):
    ''' Use implicit function theorem to compute gradient.'''
    delf_x = jnp.eye(z.shape[0])
    diag_indices = jnp.diag_indices_from(delf_x)
    delf_x = delf_x.at[diag_indices].set(1.0/jnp.cosh(linear(z, x))**2)
    delf_z = delf_x@projection
    return jax.scipy.linalg.solve(jnp.eye(z.shape[0]) - delf_z, delf_x)


key = random.PRNGKey(100)
x = random.normal(key, shape=(_VECTOR_DIM,))
print(f'Device = {x.device()}')
fixed_point = compute_fp(x)
start = time.time()
jacobian = jax.jacfwd(compute_fp)(x)
stop = time.time()
print('tol = ', jnp.linalg.norm(fixed_point - step(fixed_point, x)))
print(f'jacfwd took {stop - start} sec.')
start = time.time()
implicit_fn_jacobian = compute_gradient(fixed_point, x)
stop = time.time()
print(f'implicit gradient took {stop - start} sec.')
print('gradient err = ', jnp.linalg.norm(jacobian - implicit_fn_jacobian))
