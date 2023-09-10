import argparse
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax
import time
from tqdm import tqdm

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_DIM = 32
_EPOCH_EXAMPLES = 256000
_BATCH = 128
_LR = 1e-4
_LAMBDA = 1e-2

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
print(f'Loaded arrays from disk in {time.time() - start} secs.')

n_items = int(jnp.max(items)) + 1
n_users = items.shape[0]
key = jax.random.PRNGKey(42)
item_embeddings = None
if args.start_epoch == -1:
    print(f'Initializing embeddings for {n_items} items {n_users} users')
    item_embeddings = jax.random.normal(key, shape=(n_items, _DIM))/100
    user_embeddings = jax.random.normal(key + 1, shape=(n_users, _DIM))/100
    customer_age_vector = jax.random.normal(key + 2, shape=(_DIM,))/100
else:
    print(f'Loading embeddings from checkpoint.')
    checkpoint = jnp.load(_DATASET + f'/embeddings_{args.start_epoch}.npz')
    item_embeddings = checkpoint['item_embeddings']
    user_embeddings = checkpoint['user_embeddings']
    customer_age_vector = checkpoint['customer_age_vector']

print(items.shape, seq_lengths.shape,
      item_embeddings.shape, user_embeddings.shape,
      customer_age_vector.shape)


# Optimized (vectorized computation over all examples in batch)
##########################################################


@jax.jit
def fwd_batch_opt_core(input_item_embeddings,
                       input_user_embeddings,
                       input_customer_age_vector,
                       customer_ages_batch,
                       flat_items,
                       flat_map):
    # Modify user embeddings by input features.
    input_user_embeddings = (
        input_user_embeddings +
        jnp.expand_dims(customer_ages_batch, axis=1) *
        jnp.expand_dims(input_customer_age_vector, axis=0)
    )
    # Note: negation in next line is reversed for positive examples
    # in the following line to it.
    logits = -jnp.einsum('ij,kj->ki', input_item_embeddings,
                         input_user_embeddings)
    logits = logits.at[flat_map, flat_items].multiply(-1.0)
    nll = jnp.sum(-jnp.log(jax.nn.sigmoid(logits)), axis=-1)
    loss = jnp.sum(nll, axis=0)
    return loss


# Not jittable due to dynamic repeat.
def fwd_batch_opt(params,
                  customer_ages_batch,
                  seq_items_batch,
                  seq_lengths_batch: jnp.ndarray,
                  user_indices_batch: jnp.ndarray):
    (input_item_embeddings, input_user_embeddings,
        input_customer_age_vector) = params
    batch_size = len(seq_items_batch)
    # Flatten items from all examples into a single vector.
    flat_items = jnp.concatenate(seq_items_batch, axis=0)
    # Map back to example.
    flat_map = jnp.repeat(jnp.arange(batch_size), seq_lengths_batch)
    return fwd_batch_opt_core(input_item_embeddings,
                              input_user_embeddings[user_indices_batch],
                              input_customer_age_vector,
                              customer_ages_batch,
                              flat_items, flat_map)


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
    seq_items_batch = []
    seq_lengths_batch = []
    user_indices_batch = []
    customer_ages_batch = []
    for index in pbar:
        seq_items = items[index][:seq_lengths[index]]
        seq_items_batch.append(seq_items)
        seq_lengths_batch.append(seq_lengths[index])
        user_indices_batch.append(index)
        customer_ages_batch.append(customer_age[index])
        if len(seq_items_batch) == _BATCH:
            seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
            user_indices_batch_array = jnp.asarray(user_indices_batch)
            customer_ages_batch_array = jnp.asarray(customer_ages_batch)
            seq_lengths_batch = []
            user_indices_batch = []
            customer_ages_batch = []
            if solver_initialized:
                out_params, state = solver.update(
                    params=(item_embeddings,
                            user_embeddings,
                            customer_age_vector),
                    state=state,
                    customer_ages_batch=customer_ages_batch_array,
                    seq_items_batch=seq_items_batch,
                    seq_lengths_batch=seq_lengths_batch_array,
                    user_indices_batch=user_indices_batch_array)
                item_embeddings, user_embeddings, customer_age_vector = out_params
            else:
                state = solver.init_state(
                    init_params=(item_embeddings,
                                 user_embeddings,
                                 customer_age_vector),
                    customer_ages_batch=customer_ages_batch_array,
                    seq_items_batch=seq_items_batch,
                    seq_lengths_batch=seq_lengths_batch_array,
                    user_indices_batch=user_indices_batch_array)
                solver_initialized = True
            batches += 1
            # Update displayed loss.
            if batches % 10 == 0:
                loss = fwd_batch_opt(
                    (item_embeddings, user_embeddings, customer_age_vector),
                    customer_ages_batch_array, seq_items_batch,
                    seq_lengths_batch_array, user_indices_batch_array)
                loss = (loss/seq_lengths_batch_array.shape[0])/n_items
                if sum_loss is None:
                    sum_loss = loss
                else:
                    sum_loss += loss
                items_loss += 1
                pbar.set_description(
                    f'epoch {epoch} avg loss {sum_loss/items_loss:.4f}')
            seq_items_batch = []

    if len(seq_items_batch) > 0:
        # Use partial batch to compute loss metric only.
        seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
        user_indices_batch_array = jnp.asarray(user_indices_batch)
        customer_ages_batch_array = jnp.asarray(customer_ages_batch)
        loss = fwd_batch_opt(
            (item_embeddings, user_embeddings, customer_age_vector),
            customer_ages_batch_array, seq_items_batch,
            seq_lengths_batch_array, user_indices_batch_array)
        loss = (loss/seq_lengths_batch_array.shape[0])/n_items
        if sum_loss is None:
            sum_loss = loss
        else:
            sum_loss += loss
        items_loss += 1
    print(f'Epoch = {epoch} loss = {sum_loss/items_loss:.4f}')

    jnp.savez(_DATASET + f'/embeddings_{epoch}.npz',
              item_embeddings=item_embeddings,
              user_embeddings=user_embeddings,
              customer_age_vector=customer_age_vector)
