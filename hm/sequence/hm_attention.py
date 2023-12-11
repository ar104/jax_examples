# Implement a seq to seq transformation.
# Follows "The annotated transformer" but in jax.

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

_TRANSFORMER_EPSILON = 1e-6


class LayerNorm(NamedTuple):
    bias: jnp.ndarray
    scale: jnp.ndarray

    @classmethod
    def factory(cls, rng_key, shape: Tuple):
        rng_key, subkey1 = jax.random.split(rng_key)
        _, subkey2 = jax.random.split(rng_key)
        return LayerNorm(
            bias=jax.random.normal(subkey1, shape=shape),
            scale=jax.random.normal(subkey2, shape=shape)
        )/100

    def forward(self, input: jnp.ndarray):
        input_mean = jnp.mean(input, axis=-1, keepdims=True)
        input_sd = jnp.std(input, axis=-1, keepdims=True)
        input = (input - input_mean)(input_sd + _TRANSFORMER_EPSILON)
        return self.scale*input + self.bias


class SelfAttention(NamedTuple):
    W_K: jnp.ndarray
    W_Q: jnp.ndarray
    W_V: jnp.ndarray
    FF1: jnp.ndarray
    FF2: jnp.ndarray

    @classmethod
    def factory(cls, dim_io, num_heads):
        dim_internal = dim_io//num_heads
        # TODO: Complete this.


class Block(NamedTuple):
    '''Wraps a transformer adding dropout, norm and a residual connection'''
    norm: LayerNorm
    self_attention: SelfAttention
    rate: float

    def forward(self, usable_subkey, input: jnp.ndarray):
        attention_result = self.self_attention.forward(
            self.norm.forward(input)
        )
        keep = jax.random.bernoulli(
            usable_subkey, self.rate, attention_result.shape)
        return jnp.where(keep, attention_result/self.rate, 0)
