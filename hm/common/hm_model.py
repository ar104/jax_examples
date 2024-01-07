from typing import NamedTuple
import jax
import jax.numpy as jnp
import sys

from hm_encoder import HMEncoder

_DIM = 32
_ITEM_NET_HIDDEN_DIM = 32
_USER_NET_HIDDEN_DIM = 32
_HISTORY_NET_HIDDEN_DIM = 32


class NN(NamedTuple):
    '''Trainable weights'''
    input_layer: jnp.ndarray
    hidden_layer: jnp.ndarray
    output_layer: jnp.ndarray

    @classmethod
    def factory(self, rng_key, dim_input, dim_hidden, dim_output):
        # Generate keys to seed RNGs for parameters.
        rng_key, subkey1 = jax.random.split(rng_key)
        rng_key, subkey2 = jax.random.split(rng_key)
        _, subkey3 = jax.random.split(rng_key)
        return NN(
            input_layer=jax.random.normal(
                subkey1, shape=(dim_hidden, dim_input))/dim_input,
            hidden_layer=jax.random.normal(
                subkey2, shape=(dim_hidden, dim_hidden))/dim_hidden,
            output_layer=jax.random.normal(
                subkey3, shape=(dim_output, dim_hidden))/dim_hidden
        )


def forward_NN(nn: NN, input: jnp.ndarray) -> jnp.ndarray:
    x = jnp.einsum('bf,hf->bh', input,  nn.input_layer)
    x = jax.nn.relu(x)
    x = jnp.einsum('bh,jh->bj', x, nn.hidden_layer)
    x = jax.nn.relu(x)
    x = jnp.einsum('bj,oj->bo', x, nn.output_layer)
    return x
