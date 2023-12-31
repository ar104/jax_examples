from functools import partial
import argparse
import gc
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax
import pickle
import random
import time
from tqdm import tqdm

from hm_model import HMModel, compute_pe_matrix

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'

_EPOCH_EXAMPLES = 2560
_BATCH = 128
_LR = 0.5e-4
_LAMBDA = 1e-3
_EPSILON = 1e-6


parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='Adam', help='SGD or Adam')
parser.add_argument('--start_epoch', type=int,
                    default=-1, help='Load from epoch')
args = parser.parse_args()

print(jax.devices())


start = time.time()
data = jnp.load(_DATASET + '/tensors_history.npz')
items = data['items']
timestamps = data['timestamps']
seq_lengths = data['seq_lengths']
customer_age = data['customer_age']
articles_color_group = data['articles_color_group_name']
articles_section_name = data['articles_section_name']
articles_garment_group = data['articles_garment_group_name']
customer_fn = data['customer_fn']
customer_active = data['customer_active']
customer_club_member_status = data['customer_club_member_status']
customer_fashion_news_frequency = data['customer_fashion_news_frequency']
customer_postal_code = data['customer_postal_code']

print(f'Loaded arrays from disk in {time.time() - start} secs.')

n_articles = articles_garment_group.shape[0]
n_users = items.shape[0]
n_color_groups = int(jnp.max(articles_color_group)) + 1
n_section_names = int(jnp.max(articles_section_name)) + 1
n_garment_groups = int(jnp.max(articles_garment_group)) + 1
n_customer_club_member_status = int(jnp.max(customer_club_member_status)) + 1
n_customer_fashion_news_frequency = int(
    jnp.max(customer_fashion_news_frequency)) + 1
n_customer_postal_code = int(jnp.max(customer_postal_code)) + 1

key = jax.random.PRNGKey(42)
item_embeddings = None
if args.start_epoch == -1:
    print(f'Initializing parameters for {n_articles} items {n_users} users')
    model_parameters = HMModel.factory(
        rng_key=key,
        n_articles=n_articles,
        n_color_groups=n_color_groups,
        n_section_names=n_section_names,
        n_garment_groups=n_garment_groups,
        n_user_club_member_status=n_customer_club_member_status,
        n_user_fashion_news_frequency=n_customer_fashion_news_frequency,
        n_user_postal_code=n_customer_postal_code)
else:
    print(f'Loading embeddings from checkpoint.')
    with open(f'{_DATASET}/embeddings_{args.start_epoch}.pickle', 'rb') as f:
        model_parameters = pickle.load(f)

# Optimized (vectorized computation over all examples in batch)
##########################################################


@partial(jax.jit, static_argnames=['batch_size'])
def fwd_batch_opt_core(model_params,
                       articles_color_group,
                       articles_section_name,
                       articles_garment_group,
                       customer_ages_batch,
                       customer_fn_batch,
                       customer_active_batch,
                       customer_club_member_status_batch,
                       customer_fashion_news_frequency_batch,
                       customer_postal_code_batch,
                       flat_items,
                       flat_position_vectors,
                       flat_items_map,
                       seq_lengths_batch,
                       flat_labels,
                       flat_labels_map,
                       batch_size):
    # Modify item embeddings by input features.
    input_item_embeddings = model_params.item_embedding_vectors(
        articles_color_group=articles_color_group,
        articles_section_name=articles_section_name,
        articles_garment_group=articles_garment_group
    )
    # Modify user embeddings by input features and history.
    history_embedding_vectors = model_params.history_embedding_vectors(
        input_item_embeddings[flat_items, :] + flat_position_vectors)
    user_history_vectors = jax.ops.segment_sum(
        history_embedding_vectors,
        flat_items_map,
        num_segments=batch_size
    ) / (jnp.expand_dims(seq_lengths_batch, axis=1) + _EPSILON)
    input_user_embeddings = model_params.user_embedding_vectors(
        user_history_vectors,
        customer_ages_batch,
        customer_fn_batch,
        customer_active_batch,
        customer_club_member_status_batch,
        customer_fashion_news_frequency_batch,
        customer_postal_code_batch,
    )
    # Note: negation in next line is reversed for positive examples
    # in the following line to it.
    logits = jnp.einsum('ij,kj->ki', input_item_embeddings,
                        input_user_embeddings)
    # Compute the softmax cross entropy loss.
    log_numerator = logits[flat_labels_map, flat_labels]
    log_denominator = jax.nn.logsumexp(logits, axis=1)[flat_labels_map]
    # optimize negative log likelihood
    nll = jnp.mean(log_denominator - log_numerator)
    return nll


# Not jittable due to dynamic repeat.
def fwd_batch_opt(params: HMModel,
                  customer_ages_batch,
                  customer_fn_batch,
                  customer_active_batch,
                  customer_club_member_status_batch,
                  customer_fashion_news_frequency_batch,
                  customer_postal_code_batch,
                  articles_color_group,
                  articles_section_name,
                  articles_garment_group,
                  seq_labels_batch,
                  seq_labels_count_batch,
                  seq_history_batch,
                  seq_position_vectors_batch,
                  seq_lengths_batch: jnp.ndarray):
    batch_size = len(seq_history_batch)
    # Flatten items from all examples into a single vector.
    flat_items = jnp.concatenate(seq_history_batch, axis=0)
    flat_position_vectors = jnp.concatenate(seq_position_vectors_batch, axis=0)
    flat_items_map = jnp.repeat(jnp.arange(batch_size), seq_lengths_batch)
    flat_labels = jnp.concatenate(seq_labels_batch, axis=0)
    flat_labels_map = jnp.repeat(
        jnp.arange(batch_size), jnp.asarray(seq_labels_count_batch))
    return fwd_batch_opt_core(params,
                              # Item features.
                              articles_color_group,
                              articles_section_name,
                              articles_garment_group,
                              # User features.
                              customer_ages_batch,
                              customer_fn_batch,
                              customer_active_batch,
                              customer_club_member_status_batch,
                              customer_fashion_news_frequency_batch,
                              customer_postal_code_batch,
                              # Transactions.
                              flat_items,
                              flat_position_vectors,
                              flat_items_map,
                              seq_lengths_batch,
                              # Labels
                              flat_labels,
                              flat_labels_map,
                              # batch size - dynamic jit
                              customer_ages_batch.shape[0])


# Initialize optimizer.
if args.optimizer == 'SGD':
    opt = optax.sgd(_LR)
elif args.optimizer == 'Adam':
    opt = optax.adamw(_LR, weight_decay=_LAMBDA)
else:
    print(f'Unknown optimizer {args.optimizer}')
    exit(-1)
solver = OptaxSolver(opt=opt, fun=fwd_batch_opt)
solver_initialized = False
start_epoch = max(0, args.start_epoch)
pe_matrix = compute_pe_matrix()
for epoch in range(start_epoch, 100):
    train_indices = jax.random.choice(
        key + epoch, items.shape[0],
        (_EPOCH_EXAMPLES,))
    train_indices = jax.random.shuffle(key + epoch + 1, train_indices)
    pbar = tqdm(train_indices)
    pbar.set_description(f'epoch {epoch}')
    batches = 0
    sum_loss = None
    items_loss = 0
    seq_labels_batch = []
    seq_labels_count_batch = []
    seq_history_batch = []
    seq_position_vectors_batch = []
    seq_lengths_batch = []
    user_indices_batch = []
    customer_ages_batch = []
    customer_fn_batch = []
    customer_active_batch = []
    customer_club_member_status_batch = []
    customer_fashion_news_frequency_batch = []
    customer_postal_code_batch = []
    label_indices = jax.random.randint(
        key=key + epoch + 1,
        shape=(items.shape[0],),
        minval=0,
        maxval=seq_lengths)
    for index in pbar:
        root_timestamp = timestamps[index][label_indices[index]]
        seq_history = []
        seq_labels = []
        seq_timestamps = []
        for i in range(seq_lengths[index]):
            if timestamps[index][i] < root_timestamp:
                seq_history.append(items[index][i])
                seq_timestamps.append(root_timestamp - timestamps[index][i])
            elif (timestamps[index][i] - root_timestamp) < 16:
                seq_labels.append(items[index][i])
        seq_labels_batch.append(
            jnp.asarray([random.choice(seq_labels)], dtype=jnp.int32))
        seq_labels_count_batch.append(1)
        seq_history_batch.append(jnp.asarray(seq_history, dtype=jnp.int32))
        seq_position_vectors_batch.append(pe_matrix[seq_timestamps, :])
        seq_lengths_batch.append(len(seq_history))
        user_indices_batch.append(index)
        customer_ages_batch.append(customer_age[index])
        customer_fn_batch.append(customer_fn[index])
        customer_active_batch.append(customer_active[index])
        customer_club_member_status_batch.append(
            customer_club_member_status[index])
        customer_fashion_news_frequency_batch.append(
            customer_fashion_news_frequency[index])
        customer_postal_code_batch.append(customer_postal_code[index])
        if len(seq_history_batch) == _BATCH:
            seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
            user_indices_batch_array = jnp.asarray(user_indices_batch)
            customer_ages_batch_array = jnp.asarray(customer_ages_batch)
            customer_fn_batch_array = jnp.asarray(customer_fn_batch)
            customer_active_batch_array = jnp.asarray(customer_active_batch)
            customer_club_member_status_batch_array = jnp.asarray(
                customer_club_member_status_batch, jnp.int32)
            customer_fashion_news_frequency_batch_array = jnp.asarray(
                customer_fashion_news_frequency_batch, jnp.int32)
            customer_postal_code_batch_array = jnp.asarray(
                customer_postal_code_batch, jnp.int32)
            seq_lengths_batch = []
            user_indices_batch = []
            customer_ages_batch = []
            customer_fn_batch = []
            customer_active_batch = []
            customer_club_member_status_batch = []
            customer_fashion_news_frequency_batch = []
            customer_postal_code_batch = []
            if solver_initialized:
                model_parameters, state = solver.update(
                    params=model_parameters, state=state,
                    customer_ages_batch=customer_ages_batch_array,
                    customer_fn_batch=customer_fn_batch_array,
                    customer_active_batch=customer_active_batch_array,
                    customer_club_member_status_batch=customer_club_member_status_batch_array,
                    customer_fashion_news_frequency_batch=customer_fashion_news_frequency_batch_array,
                    customer_postal_code_batch=customer_postal_code_batch_array,
                    articles_color_group=articles_color_group,
                    articles_section_name=articles_section_name,
                    articles_garment_group=articles_garment_group,
                    seq_labels_batch=seq_labels_batch,
                    seq_labels_count_batch=seq_labels_count_batch,
                    seq_history_batch=seq_history_batch,
                    seq_position_vectors_batch=seq_position_vectors_batch,
                    seq_lengths_batch=seq_lengths_batch_array)
            else:
                state = solver.init_state(
                    init_params=model_parameters,
                    customer_ages_batch=customer_ages_batch_array,
                    customer_fn_batch=customer_fn_batch_array,
                    customer_active_batch=customer_active_batch_array,
                    customer_club_member_status_batch=customer_club_member_status_batch_array,
                    customer_fashion_news_frequency_batch=customer_fashion_news_frequency_batch_array,
                    customer_postal_code_batch=customer_postal_code_batch_array,
                    articles_color_group=articles_color_group,
                    articles_section_name=articles_section_name,
                    articles_garment_group=articles_garment_group,
                    seq_labels_batch=seq_labels_batch,
                    seq_labels_count_batch=seq_labels_count_batch,
                    seq_history_batch=seq_history_batch,
                    seq_position_vectors_batch=seq_position_vectors_batch,
                    seq_lengths_batch=seq_lengths_batch_array)
                solver_initialized = True
            batches += 1
            # Update displayed loss.
            if batches % 1 == 0:
                loss = fwd_batch_opt(
                    model_parameters,
                    customer_ages_batch_array,
                    customer_fn_batch_array,
                    customer_active_batch_array,
                    customer_club_member_status_batch_array,
                    customer_fashion_news_frequency_batch_array,
                    customer_postal_code_batch_array,
                    articles_color_group,
                    articles_section_name,
                    articles_garment_group,
                    seq_labels_batch,
                    seq_labels_count_batch,
                    seq_history_batch,
                    seq_position_vectors_batch,
                    seq_lengths_batch_array)
                if sum_loss is None:
                    sum_loss = loss
                else:
                    sum_loss += loss
                items_loss += 1
                pbar.set_description(
                    f'epoch {epoch} avg loss {sum_loss/items_loss:.4f}')
            seq_history_batch = []
            seq_position_vectors_batch = []
            seq_labels_batch = []
            seq_labels_count_batch = []
    # Avoid possible memory leak.
    jax.clear_caches()
    gc.collect()

    if len(seq_labels_batch) > 0:
        # Use partial batch to compute loss metric only.
        seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
        user_indices_batch_array = jnp.asarray(user_indices_batch)
        customer_ages_batch_array = jnp.asarray(customer_ages_batch)
        customer_fn_batch_array = jnp.asarray(customer_fn_batch)
        customer_active_batch_array = jnp.asarray(customer_active_batch)
        customer_club_member_status_batch_array = jnp.asarray(
            customer_club_member_status_batch)
        customer_fashion_news_frequency_batch_array = jnp.asarray(
            customer_fashion_news_frequency_batch)
        customer_postal_code_batch_array = jnp.asarray(
            customer_postal_code_batch)
        loss = fwd_batch_opt(
            model_parameters,
            customer_ages_batch_array,
            customer_fn_batch_array,
            customer_active_batch_array,
            customer_club_member_status_batch_array,
            customer_fashion_news_frequency_batch_array,
            customer_postal_code_batch_array,
            articles_color_group,
            articles_section_name,
            articles_garment_group,
            seq_labels_batch,
            seq_labels_count_batch,
            seq_history_batch,
            seq_position_vectors_batch,
            seq_lengths_batch_array)
        loss = (loss/seq_lengths_batch_array.shape[0])
        if sum_loss is None:
            sum_loss = loss
        else:
            sum_loss += loss
        items_loss += 1
    print(f'Epoch = {epoch} loss = {sum_loss/items_loss:.4f}')
    with open(f'{_DATASET}/embeddings_{epoch}.pickle', 'wb') as f:
        pickle.dump(model_parameters,  f)
