from typing import NamedTuple
import jax
import jax.numpy as jnp


_DIM = 32
_ITEM_NET_HIDDEN_DIM = 32
_USER_NET_HIDDEN_DIM = 32
_HISTORY_NET_HIDDEN_DIM = 32
_MAX_DAYS = 5000


def compute_pe_matrix():
    pe = jnp.zeros(shape=(_MAX_DAYS, _DIM))
    position = jnp.arange(0, _MAX_DAYS, dtype=jnp.float32)
    div_term = jnp.exp(jnp.arange(0, _DIM, 2)*(-jnp.log(10000.0))/_DIM)
    product = jnp.einsum('i,j->ij', position, div_term)
    pe = pe.at[:, 0::2].set(jnp.sin(product))
    pe = pe.at[:, 1::2].set(jnp.cos(product))
    return pe


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
    user_age_vector: jnp.ndarray
    user_net: NN
    item_embeddings: jnp.ndarray
    color_group_embeddings: jnp.ndarray
    section_name_embeddings: jnp.ndarray
    garment_group_embeddings: jnp.ndarray
    item_net: NN
    history_net: NN
    user_club_member_status_embedding: jnp.ndarray
    user_fashion_news_frequency_embedding: jnp.ndarray
    user_postal_code_embedding: jnp.ndarray
    user_fn_vector: jnp.ndarray
    user_active_vector: jnp.ndarray

    @classmethod
    def factory(cls,
                rng_key,
                n_users,
                n_articles,
                n_color_groups,
                n_section_names,
                n_garment_groups,
                n_user_club_member_status,
                n_user_fashion_news_frequency,
                n_user_postal_code):
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
            item_net=NN.factory(
                rng_key + 5, _DIM, _ITEM_NET_HIDDEN_DIM, _DIM),
            user_net=NN.factory(
                rng_key + 6, _DIM, _USER_NET_HIDDEN_DIM, _DIM),
            history_net=NN.factory(
                rng_key + 7, _DIM, _HISTORY_NET_HIDDEN_DIM, _DIM),
            user_club_member_status_embedding=jax.random.normal(
                rng_key + 14, shape=(n_user_club_member_status, _DIM)) / 100,
            user_fashion_news_frequency_embedding=jax.random.normal(
                rng_key + 15, shape=(n_user_fashion_news_frequency, _DIM)) / 100,
            user_postal_code_embedding=jax.random.normal(
                rng_key + 16, shape=(n_user_postal_code, _DIM)) / 100,
            user_fn_vector=jax.random.normal(rng_key + 17, shape=(_DIM,)) / 100,
            user_active_vector=jax.random.normal(
                rng_key + 18, shape=(_DIM,)) / 100,)

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
            jnp.expand_dims(self.user_age_vector, axis=0)
        )
        + (
            jnp.expand_dims(customer_fn_batch, axis=1) *
            jnp.expand_dims(self.user_fn_vector, axis=0)
        ) + (
            jnp.expand_dims(customer_active_batch, axis=1) *
            jnp.expand_dims(self.user_active_vector, axis=0)
        ) + (
            self.user_club_member_status_embedding
            [customer_club_member_status_batch, :]) + (
            self.user_fashion_news_frequency_embedding
            [customer_fashion_news_frequency_batch, :]) + (
            self.user_postal_code_embedding[customer_postal_code_batch, :]
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
        features = (self.color_group_embeddings[articles_color_group] +
                    self.section_name_embeddings[articles_section_name] +
                    self.garment_group_embeddings[articles_garment_group]
                    )
        transformed_features = forward_NN(self.item_net, features)
        return (self.item_embeddings + transformed_features)
