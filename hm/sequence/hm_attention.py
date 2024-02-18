# Implement a seq to seq transformation for predicting repeated purchases.
# Follows "The annotated transformer" but in jax.

import math
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

_TRANSFORMER_EPSILON = 1e-6
# Data shape = [(batch, #tokens, vector)]


class LayerNorm(NamedTuple):
    bias: jnp.ndarray
    scale: jnp.ndarray

    @classmethod
    def factory(cls, max_tokens):
        return LayerNorm(
            bias=jnp.zeros(shape=(1, max_tokens, 1), dtype=jnp.float32),
            scale=jnp.ones(shape=(1, max_tokens, 1), dtype=jnp.float32),
        )

    def forward(self, input: jnp.ndarray):
        input_mean = jnp.mean(input, axis=-1, keepdims=True)
        input_sd = jnp.std(input, axis=-1, keepdims=True)
        input = (input - input_mean)/(input_sd + _TRANSFORMER_EPSILON)
        return self.scale*input + self.bias


class SelfAttention(NamedTuple):
    W_K: jnp.ndarray  # heads * dim_io * dim_internal
    W_Q: jnp.ndarray  # heads * dim_io * dim_internal
    W_V: jnp.ndarray  # heads * dim_io * dim_internal
    W_O: jnp.ndarray  # dim_io * (heads * dim_internal)
    FF1: jnp.ndarray  # dim_io * dim_ff
    FF2: jnp.ndarray  # dim ff * dim_io

    @classmethod
    def factory(cls, rng_key, dim_io, dim_ff, num_heads):
        dim_internal = dim_io // num_heads
        assert dim_io % num_heads == 0
        rng_key, subkey1 = jax.random.split(rng_key)
        rng_key, subkey2 = jax.random.split(rng_key)
        rng_key, subkey3 = jax.random.split(rng_key)
        rng_key, subkey4 = jax.random.split(rng_key)
        rng_key, subkey5 = jax.random.split(rng_key)
        _, subkey6 = jax.random.split(rng_key)
        att_initializer = jax.nn.initializers.glorot_uniform(
            in_axis=2, out_axis=1)
        ff_initializer = jax.nn.initializers.glorot_uniform(
            in_axis=1, out_axis=0)
        output_initializer = jax.nn.initializers.glorot_uniform(
            in_axis=1, out_axis=0
        )
        return SelfAttention(
            W_K=att_initializer(
                subkey1,
                (num_heads, dim_io, dim_internal),
                dtype=jnp.float32
            ),
            W_Q=att_initializer(
                subkey2,
                (num_heads, dim_io, dim_internal),
                dtype=jnp.float32
            ),
            W_V=att_initializer(
                subkey3,
                (num_heads, dim_io, dim_internal),
                dtype=jnp.float32
            ),
            W_O=output_initializer(
                subkey4,
                (dim_io, num_heads*dim_internal),
            ),
            FF1=ff_initializer(
                subkey5,
                (dim_io, dim_ff),
                dtype=jnp.float32
            ),
            FF2=ff_initializer(
                subkey6,
                (dim_ff, dim_io),
                dtype=jnp.float32
            ),
        )

    def forward(self, input, mask):
        queries = jnp.einsum('hvi,bwv->bhwi', self.W_Q, input)
        keys = jnp.einsum('hvi,bwv->bhwi', self.W_K, input)
        values = jnp.einsum('hvi,bwv->bhwi', self.W_V, input)
        # Q,K,V of dimension batch, heads, words, inner dim.
        new_embeddings = []
        for word in range(input.shape[1]):
            logits = jnp.einsum(
                'bhi,bhwi->bhw', queries[:, :, word, :], keys[:, :, :, :])
            logits = logits / math.sqrt(queries.shape[-1])
            logits = logits + jnp.where(
                jnp.expand_dims(mask, axis=1), 0.0, -1e9
            )  # mask logits.
            weights = jax.nn.softmax(logits, axis=-1)
            weights = jnp.expand_dims(weights, axis=-1)
            transformed_embeddings = jnp.sum(
                weights*values, axis=-2, keepdims=True)
            transformed_embeddings = jnp.reshape(
                transformed_embeddings, (transformed_embeddings.shape[0], -1))
            transformed_embedding = jnp.einsum(
                'vx, bx->bv', self.W_O, transformed_embeddings)
            transformed_embedding = jnp.einsum(
                'bv, vf->bf', transformed_embedding, self.FF1)
            transformed_embedding = jax.nn.relu(transformed_embedding)
            transformed_embedding = jnp.einsum(
                'bf, fv->bv', transformed_embedding, self.FF2)
            transformed_embedding = jnp.expand_dims(
                transformed_embedding, axis=1)
            new_embeddings.append(transformed_embedding)
        return jnp.concatenate(new_embeddings, axis=1)


class Block(NamedTuple):
    '''Wraps a transformer adding dropout, norm and a residual connection'''
    norm: LayerNorm
    self_attention: SelfAttention
    rate: float

    def forward(self, usable_subkey, input: jnp.ndarray, mask: jnp.ndarray):
        attention_result = self.self_attention.forward(
            self.norm.forward(input),
            mask
        )
        keep = jax.random.bernoulli(
            usable_subkey, self.rate, attention_result.shape)
        return jnp.where(keep, attention_result/self.rate, 0)
