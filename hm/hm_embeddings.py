from functools import partial
import argparse
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax
import time
from tqdm import tqdm

from hm_model import HMModel

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'

_EPOCH_EXAMPLES = 256000
_BATCH = 128
_LR = 1e-4
_LAMBDA = 5e-1


parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='Adam', help='SGD or Adam')
parser.add_argument('--start_epoch', type=int,
                    default=-1, help='Load from epoch')
args = parser.parse_args()

print(jax.devices())


start = time.time()
data = jnp.load(_DATASET + '/tensors_history.npz')
items = data['items_dedup']
seq_lengths = data['seq_length_dedup']
customer_age = data['customer_age']
articles_color_group = data['articles_color_group_name']
articles_section_name = data['articles_section_name']
articles_garment_group = data['articles_garment_group_name']

print(f'Loaded arrays from disk in {time.time() - start} secs.')

n_articles = articles_garment_group.shape[0]
n_users = items.shape[0]
n_color_groups = int(jnp.max(articles_color_group)) + 1
n_section_names = int(jnp.max(articles_section_name)) + 1
n_garment_groups = int(jnp.max(articles_garment_group)) + 1

key = jax.random.PRNGKey(42)
item_embeddings = None
if args.start_epoch == -1:
    print(f'Initializing parameters for {n_articles} items {n_users} users')
    model_parameters = HMModel.factory(
        rng_key=key,
        n_users=n_users,
        n_articles=n_articles,
        n_color_groups=n_color_groups,
        n_section_names=n_section_names,
        n_garment_groups=n_garment_groups)
else:
    print(f'Loading embeddings from checkpoint.')
    checkpoint = jnp.load(_DATASET + f'/embeddings_{args.start_epoch}.npz')
    model_parameters = HMModel(**checkpoint)

for name, value in model_parameters._asdict().items():
    print(f'{name} {value.shape}')

# Optimized (vectorized computation over all examples in batch)
##########################################################


@partial(jax.jit, static_argnames=['batch_size'])
def fwd_batch_opt_core(model_params,
                       articles_color_group,
                       articles_section_name,
                       articles_garment_group,
                       customer_ages_batch,
                       flat_items,
                       flat_items_map,
                       seq_lengths_batch,
                       flat_labels,
                       batch_size):
    # Modify item embeddings by input features.
    input_item_embeddings = model_params.item_embedding_vectors(
        articles_color_group=articles_color_group,
        articles_section_name=articles_section_name,
        articles_garment_group=articles_garment_group
    )
    # Modify user embeddings by input features and history.
    user_history_vectors = jax.ops.segment_sum(
        input_item_embeddings[flat_items, :],
        flat_items_map,
        num_segments=batch_size
    ) / jnp.expand_dims(seq_lengths_batch, axis=1)
    input_user_embeddings = model_params.user_embedding_vectors(
        user_history_vectors, customer_ages_batch)
    # Note: negation in next line is reversed for positive examples
    # in the following line to it.
    logits = -jnp.einsum('ij,kj->ki', input_item_embeddings,
                         input_user_embeddings)
    # Compute the softmax cross entropy loss.
    log_numerator = jnp.take_along_axis(
        logits, jnp.expand_dims(flat_labels, axis=1), axis=1)
    log_denominator = jax.nn.logsumexp(logits, axis=1)
    # negative log likelihood
    loss = jnp.sum(log_denominator - jnp.squeeze(log_numerator, axis=1))
    return loss


# Not jittable due to dynamic repeat.
def fwd_batch_opt(params: HMModel,
                  customer_ages_batch,
                  articles_color_group,
                  articles_section_name,
                  articles_garment_group,
                  seq_labels_batch: jnp.ndarray,
                  seq_history_batch,
                  seq_lengths_batch: jnp.ndarray):
    batch_size = len(seq_history_batch)
    # Flatten items from all examples into a single vector.
    flat_items = jnp.concatenate(seq_history_batch, axis=0)
    flat_items_map = jnp.repeat(jnp.arange(batch_size), seq_lengths_batch)
    return fwd_batch_opt_core(params,
                              # Item features.
                              articles_color_group,
                              articles_section_name,
                              articles_garment_group,
                              # User features.
                              customer_ages_batch,
                              # Transactions.
                              flat_items,
                              flat_items_map,
                              seq_lengths_batch,
                              # Labels
                              seq_labels_batch,
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
    seq_history_batch = []
    seq_lengths_batch = []
    user_indices_batch = []
    customer_ages_batch = []
    label_indices = jax.random.randint(
        key=key + epoch + 1,
        shape=(train_indices.shape[0],),
        minval=0,
        maxval=seq_lengths[train_indices])
    for index in pbar:
        seq_history = items[index][:label_indices[index]]
        seq_label = items[index][label_indices[index]]
        seq_labels_batch.append(seq_label)
        seq_history_batch.append(seq_history)
        seq_lengths_batch.append(label_indices[index])
        user_indices_batch.append(index)
        customer_ages_batch.append(customer_age[index])
        if len(seq_history_batch) == _BATCH:
            seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
            user_indices_batch_array = jnp.asarray(user_indices_batch)
            customer_ages_batch_array = jnp.asarray(customer_ages_batch)
            seq_labels_batch_array = jnp.asarray(seq_labels_batch)
            seq_lengths_batch = []
            user_indices_batch = []
            customer_ages_batch = []
            seq_labels_batch = []
            if solver_initialized:
                model_parameters, state = solver.update(
                    params=model_parameters,
                    state=state,
                    customer_ages_batch=customer_ages_batch_array,
                    articles_color_group=articles_color_group,
                    articles_section_name=articles_section_name,
                    articles_garment_group=articles_garment_group,
                    seq_labels_batch=seq_labels_batch_array,
                    seq_history_batch=seq_history_batch,
                    seq_lengths_batch=seq_lengths_batch_array)
            else:
                state = solver.init_state(
                    init_params=model_parameters,
                    customer_ages_batch=customer_ages_batch_array,
                    articles_color_group=articles_color_group,
                    articles_section_name=articles_section_name,
                    articles_garment_group=articles_garment_group,
                    seq_labels_batch=seq_labels_batch_array,
                    seq_history_batch=seq_history_batch,
                    seq_lengths_batch=seq_lengths_batch_array)
                solver_initialized = True
            batches += 1
            # Update displayed loss.
            if batches % 10 == 0:
                loss = fwd_batch_opt(
                    model_parameters,
                    customer_ages_batch_array,
                    articles_color_group,
                    articles_section_name,
                    articles_garment_group,
                    seq_labels_batch_array,
                    seq_history_batch,
                    seq_lengths_batch_array)
                loss = (loss/seq_lengths_batch_array.shape[0])
                if sum_loss is None:
                    sum_loss = loss
                else:
                    sum_loss += loss
                items_loss += 1
                pbar.set_description(
                    f'epoch {epoch} avg loss {sum_loss/items_loss:.4f}')
            seq_history_batch = []

    if len(seq_labels_batch) > 0:
        # Use partial batch to compute loss metric only.
        seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
        user_indices_batch_array = jnp.asarray(user_indices_batch)
        customer_ages_batch_array = jnp.asarray(customer_ages_batch)
        seq_labels_batch_array = jnp.asarray(seq_labels_batch)
        loss = fwd_batch_opt(
            model_parameters,
            customer_ages_batch_array,
            articles_color_group,
            articles_section_name,
            articles_garment_group,
            seq_labels_batch_array,
            seq_history_batch,
            seq_lengths_batch_array)
        loss = (loss/seq_lengths_batch_array.shape[0])
        if sum_loss is None:
            sum_loss = loss
        else:
            sum_loss += loss
        items_loss += 1
    print(f'Epoch = {epoch} loss = {sum_loss/items_loss:.4f}')

    jnp.savez(
        _DATASET + f'/embeddings_{epoch}.npz', **model_parameters._asdict())
