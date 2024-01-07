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


class HMModel(NamedTuple):
    '''Holds all trainable weights.'''
    encoder: HMEncoder
    user_net: NN
    item_net: NN
    history_net: NN

    @classmethod
    def factory(cls,
                rng_key,
                n_articles,
                n_color_groups,
                n_section_names,
                n_garment_groups,
                n_user_club_member_status,
                n_user_fashion_news_frequency,
                n_user_postal_code):
        '''Constructs and returns initialized model parameters'''
        rng_key, encoder_rng_key = jax.random.split(rng_key)
        rng_key, user_rng_key = jax.random.split(rng_key)
        rng_key, item_rng_key = jax.random.split(rng_key)
        rng_key, history_rng_key = jax.random.split(rng_key)
        return HMModel(
            user_net=NN.factory(
                user_rng_key, _DIM, _USER_NET_HIDDEN_DIM, _DIM),
            item_net=NN.factory(
                item_rng_key, _DIM, _ITEM_NET_HIDDEN_DIM, _DIM),
            history_net=NN.factory(
                history_rng_key, _DIM, _HISTORY_NET_HIDDEN_DIM, _DIM),
            encoder=HMEncoder.factory(encoder_rng_key,
                                      _DIM,
                                      n_articles,
                                      n_color_groups,
                                      n_section_names,
                                      n_garment_groups,
                                      n_user_club_member_status,
                                      n_user_fashion_news_frequency,
                                      n_user_postal_code),
        )

    def history_embedding_vectors(self,
                                  batch_history_vectors):
        '''Computes the history embedding vectors.'''
        return forward_NN(self.history_net, batch_history_vectors)

    def user_embedding_vectors(self,
                               batch_user_history_vectors,
                               batch_user_ages,
                               customer_fn_batch,
                               customer_active_batch,
                               customer_club_member_status_batch,
                               customer_fashion_news_frequency_batch,
                               customer_postal_code_batch,
                               skip=True):
        '''Computes the user embedding vectors.'''
        features = batch_user_history_vectors + (
            jnp.expand_dims(batch_user_ages, axis=1) *
            jnp.expand_dims(self.encoder.user_age_vector, axis=0)
        )
        + (
            jnp.expand_dims(customer_fn_batch, axis=1) *
            jnp.expand_dims(self.encoder.user_fn_vector, axis=0)
        ) + (
            jnp.expand_dims(customer_active_batch, axis=1) *
            jnp.expand_dims(self.encoder.user_active_vector, axis=0)
        ) + (
            self.encoder.user_club_member_status_embedding
            [customer_club_member_status_batch, :]) + (
            self.encoder.user_fashion_news_frequency_embedding
            [customer_fashion_news_frequency_batch, :]) + (
            self.encoder.user_postal_code_embedding[
                customer_postal_code_batch, :]
        )
        transformed_features = forward_NN(self.user_net, features)
        if skip:
            return transformed_features + batch_user_history_vectors
        else:
            return transformed_features

    def item_embedding_vectors(self,
                               articles_color_group,
                               articles_section_name,
                               articles_garment_group):
        '''Computes the item embedding vectors.'''
        features = (self.encoder.color_group_embeddings[articles_color_group] +
                    self.encoder.section_name_embeddings[articles_section_name] +
                    self.encoder.garment_group_embeddings[articles_garment_group]
                    )
        transformed_features = forward_NN(self.item_net, features)
        return (self.encoder.item_embeddings + transformed_features)
