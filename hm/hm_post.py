from collections import defaultdict
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm
import time

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_EMBEDDINGS = 'C:\\Users\\aroym\\OneDrive\\Documents\\GitHub\\jax_examples\\hm\\embeddings_23.npz'
_HISTORY = 64
_K = 12
_BATCH = 4096


@jax.jit
def get_topk(history, embeddings):
    user_embedding = jnp.mean(history, axis=0)
    scores = embeddings @ jnp.transpose(user_embedding)
    topk = jnp.argsort(-scores)[:_K]
    return topk


@jax.jit
def fwd_batch_opt_core(
        input_embeddings, flat_items, flat_map, seq_lengths_batch):
    flat_item_embeddings = input_embeddings[flat_items]
    # Compute user embedding as mean of all purchased item embeddings.
    user_embeddings = jax.ops.segment_sum(
        flat_item_embeddings, flat_map, num_segments=_BATCH,
        indices_are_sorted=True)
    user_embeddings /= jnp.expand_dims(seq_lengths_batch, axis=-1)
    logits = jnp.einsum('ij,kj->ki', input_embeddings, user_embeddings)
    _, topk = jax.lax.top_k(logits, _K)
    return topk


# Not jittable due to dynamic repeat.
def topk_batch_opt(
        input_embeddings, seq_items_batch, seq_lengths_batch: jnp.ndarray):
    batch_size = len(seq_items_batch)
    # Flatten items from all examples into a single vector.
    flat_items = jnp.concatenate(seq_items_batch, axis=0)
    # Map back to example.
    flat_map = jnp.repeat(jnp.arange(batch_size), seq_lengths_batch)
    return fwd_batch_opt_core(
        input_embeddings, flat_items, flat_map, seq_lengths_batch)


def process_batch(
        embeddings, seq_items_batch, seq_lengths_batch, cid_batch, predictions):
    if len(cid_batch) == _BATCH:
        topk_batch = topk_batch_opt(
            embeddings, seq_items_batch, jnp.asarray(seq_lengths_batch))
        topk_batch = topk_batch.tolist()
        # for cid, topk_list in zip(cid_batch, topk_batch):
        #    topk = [str(mapping[e]) for e in topk_list]
        #    predictions.write(cid + ',' + ' '.join(topk) + '\n')
    else:
        for cid, history in zip(cid_batch, seq_items_batch):
            topk_list = get_topk(embeddings[history], embeddings)
            topk = [str(mapping[e.item()]) for e in topk_list]
            predictions.write(cid + ',' + ' '.join(topk) + '\n')


start = time.time()
data = jnp.load(_DATASET + '/tensors_history.npz')
items = data['items']
seq_lengths = data['seq_lengths']
embeddings = jnp.load(_EMBEDDINGS)['item_embeddings']
mapping = pd.read_csv(_DATASET + '\\item_map.csv')
cid_map = pd.read_csv(_DATASET + '\\cid_map.csv')
cid_map = cid_map['customer_id'].to_list()
all_customers = set(pd.read_csv(
    _DATASET + '\\customers.csv')['customer_id'].to_list())
print(f'Loaded from disk in {time.time() - start} secs.')
mapping.set_index('enum', inplace=True)
mapping = mapping['article_id'].to_dict()
print(embeddings.shape)
print(items.shape)
print(len(mapping.keys()))
n_embeddings = embeddings.shape[0]
pbar = tqdm(cid_map)
pbar.set_description('KNN Search')
item_freq = defaultdict(lambda: 0)
with open(_DATASET + '\\predictions.csv', 'w') as predictions:
    predictions.write('customer_id,prediction\n')
    seq_items_batch = []
    seq_lengths_batch = []
    cid_batch = []
    for index, cid in enumerate(pbar):
        history_length = min(seq_lengths[index], _HISTORY)
        item_history = items[index][-history_length:]
        seq_items_batch.append(item_history)
        seq_lengths_batch.append(history_length)
        cid_batch.append(cid)
        if len(seq_items_batch) == _BATCH:
            process_batch(embeddings, seq_items_batch,
                          seq_lengths_batch, cid_batch, predictions)
            seq_lengths_batch = []
            seq_items_batch = []
            cid_batch = []
    if len(cid_batch) > 0:
        process_batch(embeddings, seq_items_batch,
                      seq_lengths_batch, cid_batch, predictions)
    missing_customers = all_customers.difference(cid_map)
    global_top_k = list(item_freq.items())
    global_top_k.sort(key=lambda e: e[1], reverse=True)
    global_top_k = global_top_k[:_K]
    global_top_k = [e[0] for e in global_top_k]
    pbar = tqdm(missing_customers)
    pbar.set_description('Missing customers')
    for cid in pbar:
        predictions.write(cid + ',' + ' '.join(global_top_k) + '\n')
