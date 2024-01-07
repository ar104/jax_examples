from typing import NamedTuple
import jax
import jax.numpy as jnp

_MAX_DAYS = 5000


class HMEncoder(NamedTuple):
    user_age_vector: jnp.ndarray
    item_embeddings: jnp.ndarray
    color_group_embeddings: jnp.ndarray
    section_name_embeddings: jnp.ndarray
    garment_group_embeddings: jnp.ndarray
    user_club_member_status_embedding: jnp.ndarray
    user_fashion_news_frequency_embedding: jnp.ndarray
    user_postal_code_embedding: jnp.ndarray
    user_fn_vector: jnp.ndarray
    user_active_vector: jnp.ndarray

    @classmethod
    def factory(cls,
                rng_key,
                dim,
                n_articles,
                n_color_groups,
                n_section_names,
                n_garment_groups,
                n_user_club_member_status,
                n_user_fashion_news_frequency,
                n_user_postal_code):
        '''Constructs and returns initialized feature encoder.'''
        split_keys = []
        for _ in range(10):
            rng_key, subkey = jax.random.split(rng_key)
            split_keys.append(subkey)
        return HMEncoder(
            user_age_vector=jax.random.normal(
                split_keys[0], shape=(dim,)) / 100,
            item_embeddings=jax.random.normal(
                split_keys[1], shape=(n_articles, dim)) / 1000,
            color_group_embeddings=jax.random.normal(
                split_keys[2], shape=(n_color_groups, dim)) / 100,
            section_name_embeddings=jax.random.normal(
                split_keys[3], shape=(n_section_names, dim)) / 100,
            garment_group_embeddings=jax.random.normal(
                split_keys[4], shape=(n_garment_groups, dim)) / 100,
            user_club_member_status_embedding=jax.random.normal(
                split_keys[5], shape=(n_user_club_member_status, dim)) / 100,
            user_fashion_news_frequency_embedding=jax.random.normal(
                split_keys[6], shape=(n_user_fashion_news_frequency, dim)) / 100,
            user_postal_code_embedding=jax.random.normal(
                split_keys[7], shape=(n_user_postal_code, dim)) / 100,
            user_fn_vector=jax.random.normal(split_keys[8], shape=(dim,)) / 100,
            user_active_vector=jax.random.normal(
                split_keys[9], shape=(dim,)) / 100,
        )

    def user_embedding(self,
                       batch_user_ages,
                       customer_fn_batch,
                       customer_active_batch,
                       customer_club_member_status_batch,
                       customer_fashion_news_frequency_batch,
                       customer_postal_code_batch):
        '''Embeds a user using only the user features.'''
        return (
            jnp.expand_dims(batch_user_ages, axis=1) *
            jnp.expand_dims(self.user_age_vector, axis=0)
        ) + (
            jnp.expand_dims(customer_fn_batch, axis=1) *
            jnp.expand_dims(self.user_fn_vector, axis=0)
        ) + (
            jnp.expand_dims(customer_active_batch, axis=1) *
            jnp.expand_dims(self.user_active_vector, axis=0)
        ) + (
            self.user_club_member_status_embedding
            [customer_club_member_status_batch, :]
        ) + (
            self.user_fashion_news_frequency_embedding
            [customer_fashion_news_frequency_batch, :]
        ) + (
            self.user_postal_code_embedding[customer_postal_code_batch, :]
        )


def compute_pe_matrix(dim):
    pe = jnp.zeros(shape=(_MAX_DAYS, dim))
    position = jnp.arange(0, _MAX_DAYS, dtype=jnp.float32)
    div_term = jnp.exp(jnp.arange(0, dim, 2)*(-jnp.log(10000.0))/dim)
    product = jnp.einsum('i,j->ij', position, div_term)
    pe = pe.at[:, 0::2].set(jnp.sin(product))
    pe = pe.at[:, 1::2].set(jnp.cos(product))
    return pe
