import argparse
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax
import time
from tqdm import tqdm

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_DIM = 32
_EPOCH_EXAMPLES = 1024000
_BATCH = 4096
_LR = 1e-4
_LAMBDA = 1e-8
_EPSILON = 1e-10

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
print(f'Loaded arrays from disk in {time.time() - start} secs.')

n_items = int(jnp.max(items)) + 1
key = jax.random.PRNGKey(42)
item_embeddings = None
if args.start_epoch == -1:
    print(f'Initializing embeddings for {n_items} items')
    item_embeddings = jax.random.normal(key, shape=(n_items, _DIM))/100
else:
    print(f'Loading embeddings from checkpoint.')
    checkpoint = jnp.load(_DATASET + f'/embeddings_{args.start_epoch}.npz')
    item_embeddings = checkpoint['item_embeddings']
print(items.shape, seq_lengths.shape, item_embeddings.shape)

# Unoptimized (iterate over examples in batch)
#########################################################


@jax.jit
def fwd(input_embeddings, seq_items):
    user_embedding = jnp.mean(input_embeddings[seq_items], axis=0)
    probs = jax.nn.sigmoid(input_embeddings @ jnp.transpose(user_embedding))
    truth = jnp.zeros(shape=probs.shape).at[seq_items].set(1.0)
    nll = -jnp.sum(jnp.log(jnp.where(truth, probs, 1.0 - probs) + _EPSILON))
    return nll


def fwd_batch(input_embeddings, seq_items_batch):
    batch_size = len(seq_items_batch)
    return sum(
        [fwd(input_embeddings,
             seq_items_batch[i]) for i in range(batch_size)])


def train_batch(seq_items_batch):
    global item_embeddings
    loss = fwd_batch(item_embeddings, seq_items_batch)
    grad_loss = grad_fwd_batch(item_embeddings, seq_items_batch)
    item_embeddings = (1.0 - _LAMBDA)*item_embeddings - _LR*grad_loss
    return (loss/len(seq_items_batch))/n_items


grad_fwd_batch = jax.grad(fwd_batch, argnums=0)


# Optimized (vectorized computation over all examples in batch)
##########################################################


@jax.jit
def fwd_batch_opt_core(
        input_embeddings, flat_items, flat_map, seq_lengths_batch):
    flat_item_embeddings = input_embeddings[flat_items]
    # Compute user embedding as mean of all purchased item embeddings.
    user_embeddings = jax.ops.segment_sum(
        flat_item_embeddings, flat_map, num_segments=_BATCH,
        indices_are_sorted=True)
    user_embeddings /= jnp.expand_dims(seq_lengths_batch, axis=-1)
    # Note: negation in next line is reversed for positive examples
    # in the following line.
    logits = -jnp.einsum('ij,kj->ki', input_embeddings, user_embeddings)
    logits = logits.at[flat_map, flat_items].multiply(-1.0)
    nll = jnp.sum(-jnp.log(jax.nn.sigmoid(logits)), axis=-1)
    loss = jnp.sum(nll, axis=0)
    return loss


# Not jittable due to dynamic repeat.
def fwd_batch_opt(
        input_embeddings, seq_items_batch, seq_lengths_batch: jnp.ndarray):
    batch_size = len(seq_items_batch)
    # Flatten items from all examples into a single vector.
    flat_items = jnp.concatenate(seq_items_batch, axis=0)
    # Map back to example.
    flat_map = jnp.repeat(jnp.arange(batch_size), seq_lengths_batch)
    return fwd_batch_opt_core(
        input_embeddings, flat_items, flat_map, seq_lengths_batch)


train_indices = jax.random.choice(key, items.shape[0], (_EPOCH_EXAMPLES,))

# Initialize optimizer.
if args.optimizer == 'SGD':
    opt = optax.sgd(_LR)
elif args.optimizer == 'Adam':
    opt = optax.adam(_LR)
else:
    print(f'Unknown optimizer {args.optimizer}')
    exit(-1)
solver = OptaxSolver(opt=opt, fun=fwd_batch_opt)
solver_initialized = False

start_epoch = max(0, args.start_epoch)
for epoch in range(start_epoch, 100):
    pbar = tqdm(train_indices)
    pbar.set_description(f'epoch {epoch}')
    batches = 0
    sum_loss = None
    items_loss = 0
    seq_items_batch = []
    seq_lengths_batch = []
    for index in pbar:
        seq_items = items[index][:seq_lengths[index]]
        seq_items_batch.append(seq_items)
        seq_lengths_batch.append(seq_lengths[index])
        if len(seq_items_batch) == _BATCH:
            seq_lengths_batch_array = jnp.asarray(seq_lengths_batch)
            seq_lengths_batch = []
            if solver_initialized:
                item_embeddings, state = solver.update(
                    params=item_embeddings,
                    state=state,
                    seq_items_batch=seq_items_batch,
                    seq_lengths_batch=seq_lengths_batch_array)
            else:
                state = solver.init_state(
                    init_params=item_embeddings,
                    seq_items_batch=seq_items_batch,
                    seq_lengths_batch=seq_lengths_batch_array)
                solver_initialized = True
            batches += 1
            # Update displayed loss.
            if batches % 10 == 0:
                loss = fwd_batch_opt(
                    item_embeddings, seq_items_batch, seq_lengths_batch_array)
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
        loss = fwd_batch_opt(
            item_embeddings, seq_items_batch, seq_lengths_batch_array)
        loss = (loss/seq_lengths_batch_array.shape[0])/n_items
        if sum_loss is None:
            sum_loss = loss
        else:
            sum_loss += loss
        items_loss += 1
    print(f'Epoch = {epoch} loss = {sum_loss/items_loss:.4f}')

    jnp.savez(_DATASET + f'/embeddings_{epoch}.npz',
              item_embeddings=item_embeddings)
