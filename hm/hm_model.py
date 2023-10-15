from typing import NamedTuple
import jax
import jax.numpy as jnp


_DIM = 32
_ITEM_NET_HIDDEN_DIM = 32
_USER_NET_HIDDEN_DIM = 32
_HISTORY_NET_HIDDEN_DIM = 32


class HMModel(NamedTuple):
    '''Holds all trainable weights.'''
    user_age_vector: jnp.ndarray
    user_net_input_layer: jnp.ndarray
    user_net_hidden_layer: jnp.ndarray
    user_net_output_layer: jnp.ndarray
    item_embeddings: jnp.ndarray
    color_group_embeddings: jnp.ndarray
    section_name_embeddings: jnp.ndarray
    garment_group_embeddings: jnp.ndarray
    item_net_input_layer: jnp.ndarray
    item_net_hidden_layer: jnp.ndarray
    item_net_output_layer: jnp.ndarray
    history_net_input_layer: jnp.ndarray
    history_net_hidden_layer: jnp.ndarray
    history_net_output_layer: jnp.ndarray

    @classmethod
    def factory(cls,
                rng_key,
                n_users,
                n_articles,
                n_color_groups,
                n_section_names,
                n_garment_groups):
        '''Constructs and returns initialized model parameters'''
        return HMModel(
            user_age_vector=jax.random.normal(rng_key + 1, shape=(_DIM,)) / 100,
            item_embeddings=jax.random.normal(
                rng_key + 2, shape=(n_articles, _DIM)) / 1000,
            color_group_embeddings=jax.random.normal(
                rng_key + 3, shape=(n_color_groups, _DIM)) / 100,
            section_name_embeddings=jax.random.normal(
                rng_key + 4, shape=(n_section_names, _DIM)) / 100,
            garment_group_embeddings=jax.random.normal(
                rng_key + 4, shape=(n_garment_groups, _DIM)) / 100,
            item_net_input_layer=jax.random.normal(
                rng_key + 5, shape=(_ITEM_NET_HIDDEN_DIM, _DIM))/_DIM,
            item_net_hidden_layer=jax.random.normal(
                rng_key + 6,
                shape=(_ITEM_NET_HIDDEN_DIM, _ITEM_NET_HIDDEN_DIM))/_ITEM_NET_HIDDEN_DIM,
            item_net_output_layer=jax.random.normal(
                rng_key + 7,
                shape=(_DIM, _ITEM_NET_HIDDEN_DIM))/_ITEM_NET_HIDDEN_DIM,
            user_net_input_layer=jax.random.normal(
                rng_key + 8, shape=(_USER_NET_HIDDEN_DIM, _DIM))/_DIM,
            user_net_hidden_layer=jax.random.normal(
                rng_key + 9,
                shape=(_USER_NET_HIDDEN_DIM, _USER_NET_HIDDEN_DIM))/_USER_NET_HIDDEN_DIM,
            user_net_output_layer=jax.random.normal(
                rng_key + 10,
                shape=(_DIM, _USER_NET_HIDDEN_DIM))/_USER_NET_HIDDEN_DIM,
            history_net_input_layer=jax.random.normal(
                rng_key + 11, shape=(_HISTORY_NET_HIDDEN_DIM, _DIM))/_DIM,
            history_net_hidden_layer=jax.random.normal(
                rng_key + 12,
                shape=(_HISTORY_NET_HIDDEN_DIM, _HISTORY_NET_HIDDEN_DIM))/_HISTORY_NET_HIDDEN_DIM,
            history_net_output_layer=jax.random.normal(
                rng_key + 13,
                shape=(_DIM, _HISTORY_NET_HIDDEN_DIM))/_HISTORY_NET_HIDDEN_DIM,
        )

    def history_embedding_vectors(self,
                                  batch_history_vectors):
        '''Computes the history embedding vectors.'''
        transformed_features = jnp.einsum(
            'bf,hf->bh', batch_history_vectors, self.history_net_input_layer)
        transformed_features = jax.nn.relu(transformed_features)
        transformed_features = jnp.einsum(
            'bi,ij->bj', transformed_features, self.history_net_hidden_layer)
        transformed_features = jax.nn.relu(transformed_features)
        transformed_features = jnp.einsum(
            'bi,io->bo', transformed_features, self.history_net_output_layer)
        return transformed_features

    def user_embedding_vectors(self,
                               batch_user_history_vectors,
                               batch_user_ages, skip=True):
        '''Computes the user embedding vectors.'''
        features = (
            jnp.expand_dims(batch_user_ages, axis=1) *
            jnp.expand_dims(self.user_age_vector, axis=0)
        ) + batch_user_history_vectors
        transformed_features = jnp.einsum(
            'bf,hf->bh', features, self.user_net_input_layer)
        transformed_features = jax.nn.relu(transformed_features)
        transformed_features = jnp.einsum(
            'bi,ij->bj', transformed_features, self.user_net_hidden_layer)
        transformed_features = jax.nn.relu(transformed_features)
        transformed_features = jnp.einsum(
            'bi,io->bo', transformed_features, self.user_net_output_layer)
        if skip:
            return transformed_features + batch_user_history_vectors
        else:
            return transformed_features

    def item_embedding_vectors(self,
                               articles_color_group,
                               articles_section_name,
                               articles_garment_group):
        '''Computes the item embedding vectors.'''
        features = (self.color_group_embeddings[articles_color_group] +
                    self.section_name_embeddings[articles_section_name] +
                    self.garment_group_embeddings[articles_garment_group]
                    )
        transformed_features = jnp.einsum(
            'bf,hf->bh', features, self.item_net_input_layer)
        transformed_features = jax.nn.relu(transformed_features)
        transformed_features = jnp.einsum(
            'bi,ij->bj', transformed_features, self.item_net_hidden_layer)
        transformed_features = jax.nn.relu(transformed_features)
        transformed_features = jnp.einsum(
            'bi,io->bo', transformed_features, self.item_net_output_layer)
        return (self.item_embeddings + transformed_features)
