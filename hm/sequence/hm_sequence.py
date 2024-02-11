import jax
import hm_attention
import sys
from typing import NamedTuple

sys.path.append('../common')    # noqa
from hm_model import (
    _DIM,
    _ITEM_NET_HIDDEN_DIM,
    _USER_NET_HIDDEN_DIM,
    _REPURCHASE_NET_HIDDEN_DIM,
    NN,
    forward_NN,
)   # noqa
from hm_encoder import HMEncoder, compute_pe_matrix    # noqa

_HEADS = 1
_RATE = 0.3


def create_attention_block(rng_key, seq_length):
    object_creation_key, rng_key = jax.random.split(rng_key)
    attention_block = hm_attention.SelfAttention.factory(
        object_creation_key, dim_io=_DIM, dim_ff=_DIM, num_heads=_HEADS)
    norm = hm_attention.LayerNorm.factory(max_tokens=seq_length)
    return hm_attention.Block(
        norm=norm, self_attention=attention_block, rate=_RATE)


class HMModel(NamedTuple):
    '''Holds all trainable weights.'''
    encoder: HMEncoder
    user_net: NN
    item_net: NN
    attention_block: hm_attention.Block
    repurchase_net: NN

    @classmethod
    def factory(cls,
                rng_key,
                n_articles,
                n_color_groups,
                n_section_names,
                n_garment_groups,
                n_user_club_member_status,
                n_user_fashion_news_frequency,
                n_user_postal_code,
                max_seq_length):
        '''Constructs and returns initialized model parameters'''
        rng_key, encoder_rng_key = jax.random.split(rng_key)
        rng_key, user_rng_key = jax.random.split(rng_key)
        rng_key, item_rng_key = jax.random.split(rng_key)
        rng_key, attention_rng_key = jax.random.split(rng_key)
        _, repurchase_net_key = jax.random.split(rng_key)
        return HMModel(
            user_net=NN.factory(
                user_rng_key, _DIM, _USER_NET_HIDDEN_DIM, _DIM),
            item_net=NN.factory(
                item_rng_key, _DIM, _ITEM_NET_HIDDEN_DIM, _DIM),
            attention_block=create_attention_block(
                attention_rng_key, max_seq_length),
            repurchase_net=NN.factory(
                repurchase_net_key, _DIM, _REPURCHASE_NET_HIDDEN_DIM, 1),
            encoder=HMEncoder.factory(
                encoder_rng_key, _DIM, n_articles, n_color_groups,
                n_section_names, n_garment_groups, n_user_club_member_status,
                n_user_fashion_news_frequency, n_user_postal_code),)

    def user_embedding_vectors(self,
                               batch_user_ages,
                               customer_fn_batch,
                               customer_active_batch,
                               customer_club_member_status_batch,
                               customer_fashion_news_frequency_batch,
                               customer_postal_code_batch):
        user_embedding = self.encoder.user_embedding(
            batch_user_ages,
            customer_fn_batch,
            customer_active_batch,
            customer_club_member_status_batch,
            customer_fashion_news_frequency_batch,
            customer_postal_code_batch,
        )
        transformed_features = forward_NN(self.user_net, user_embedding)
        return transformed_features

    def attention_embedding_vectors(self,
                                    rng_key,
                                    batch_history_vectors):
        '''Computes the attention embedding vectors.'''
        return self.attention_block.forward(rng_key, batch_history_vectors)

    def repurchase_logits(self, user_embeddings, sequence_vectors):
        '''Computes logits for repurchase probabilities'''
        return forward_NN(
            self.repurchase_net,
            jax.numpy.expand_dims(user_embeddings, axis=1) + sequence_vectors
        )

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
