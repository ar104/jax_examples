from collections import defaultdict
import pandas as pd
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
import time

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_EMBEDDINGS = 'C:\\Users\\aroym\\OneDrive\\Documents\\GitHub\\jax_examples\\hm\\embeddings_23.npz'
_HISTORY = 64
_K = 12


@jit
def get_topk(history, embeddings):
    user_embedding = jnp.mean(history, axis=0)
    scores = embeddings @ jnp.transpose(user_embedding)
    topk = jnp.argsort(-scores)[:_K]
    return topk


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
    for index, cid in enumerate(pbar):
        history_length = min(seq_lengths[index], _HISTORY)
        item_history = items[index][:-history_length]
        topk = get_topk(embeddings[item_history], embeddings)
        topk = [str(mapping[e.item()]) for e in topk]
        predictions.write(cid + ',' + ' '.join(topk) + '\n')
    missing_customers = all_customers.difference(cid_map)
    global_top_k = list(item_freq.items())
    global_top_k.sort(key=lambda e: e[1], reverse=True)
    global_top_k = global_top_k[:_K]
    global_top_k = [e[0] for e in global_top_k]
    pbar = tqdm(missing_customers)
    pbar.set_description('Missing customers')
    for cid in pbar:
        predictions.write(cid + ',' + ' '.join(global_top_k) + '\n')
