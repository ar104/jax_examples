from typing import NamedTuple
import jax
import jax.numpy as jnp
import sys

sys.path.append('../common')    # noqa
from hm_model import (
    _DIM,
    _ITEM_NET_HIDDEN_DIM,
    _USER_NET_HIDDEN_DIM,
    _HISTORY_NET_HIDDEN_DIM,
    NN,
    forward_NN,
)   # noqa
from hm_encoder import HMEncoder, compute_pe_matrix    # noqa


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

    def user_embedding_vectors(self,
                               batch_user_history_vectors,
                               batch_user_ages,
                               customer_fn_batch,
                               customer_active_batch,
                               customer_club_member_status_batch,
                               customer_fashion_news_frequency_batch,
                               customer_postal_code_batch,
                               skip=True):
        user_embedding = self.encoder.user_embedding(
            batch_user_ages,
            customer_fn_batch,
            customer_active_batch,
            customer_club_member_status_batch,
            customer_fashion_news_frequency_batch,
            customer_postal_code_batch,
        )
        transformed_features = forward_NN(
            self.user_net, user_embedding + batch_user_history_vectors)
        if skip:
            return transformed_features + batch_user_history_vectors
        else:
            return transformed_features

    def history_embedding_vectors(self,
                                  batch_history_vectors):
        '''Computes the history embedding vectors.'''
        return forward_NN(self.history_net, batch_history_vectors)

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
