import jax
import jax.numpy as jnp
import time
from tqdm import tqdm

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_DIM = 32
_EPOCH_EXAMPLES = 320
_BATCH = 32
_LR = 1e-3
_LAMBDA = 1e-8
_EPSILON = 1e-10


start = time.time()
data = jnp.load(_DATASET + '\\tensors_history.npz')
items = data['items']
seq_lengths = data['seq_lengths']
print(f'Loaded arrays from disk in {time.time() - start} secs.')

n_items = int(jnp.max(items))
print(f'Initializing embeddings for {n_items} items')
key = jax.random.PRNGKey(42)
item_embeddings = jax.random.normal(key, shape=(n_items, _DIM))/100
print(items.shape, seq_lengths.shape)


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


grad_fwd_batch = jax.grad(fwd_batch, argnums=0)


def train_batch(seq_items_batch):
    global item_embeddings
    loss = fwd_batch(item_embeddings, seq_items_batch)
    grad_loss = grad_fwd_batch(item_embeddings, seq_items_batch)
    item_embeddings = (1.0 - _LAMBDA)*item_embeddings - _LR*grad_loss
    return (loss/len(seq_items_batch))/n_items


for epoch in range(5):
    train_indices = jax.random.choice(key, n_items, (_EPOCH_EXAMPLES,))
    pbar = tqdm(train_indices)
    pbar.set_description(f'epoch {epoch}')
    batches = 0
    avg_loss = None
    seq_items_batch = []
    for index in pbar:
        seq_items = items[index][:seq_lengths[index]]
        seq_items_batch.append(seq_items)
        if len(seq_items_batch) == _BATCH:
            loss = train_batch(seq_items_batch)
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = 0.8*avg_loss + 0.2*loss
            batches += 1
            seq_items_batch = []
            if batches % 1 == 0:
                pbar.set_description(f'epoch {epoch} avg loss {avg_loss:.4f}')

    if len(seq_items_batch) > 0:
        loss = train_batch(seq_items_batch)
        if avg_loss is None:
            avg_loss = loss
        else:
            avg_loss = 0.8*avg_loss + 0.2*loss
    print(f'Epoch = {epoch} loss = {avg_loss:.4f}')

jnp.savez(_DATASET + '//embeddings.npz', item_embeddings=item_embeddings)
