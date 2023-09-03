import argparse
from collections import defaultdict
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm
import time

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_EMBEDDINGS = _DATASET + '/embeddings_test.npz'
_HISTORY = 64
_K = 12
_BATCH = 4096

parser = argparse.ArgumentParser()
parser.add_argument('--use_duplicate_features', action='store_true')
args = parser.parse_args()


@jax.jit
def topk_batch_opt(user_embeddings, item_embeddings):
    logits = jnp.einsum('ij,kj->ki', item_embeddings, user_embeddings)
    _, topk = jax.lax.top_k(logits, _K)
    return topk


def metrics(pred, truth):
    truth_set = set(truth)
    ap_list = []
    hits = 0
    for i in range(len(pred)):
        if pred[i] in truth_set:
            hits += 1
            ap_list.append(hits/(1 + i))
        else:
            ap_list.append(0.0)
    precision = hits/len(pred)
    ap = sum(ap_list)/min(len(truth), len(pred))
    return precision, ap


def process_batch(user_embeddings, item_embeddings, seq_items_batch,
                  seq_lengths_batch, cid_batch, predictions):
    precisions = []
    aps = []
    topk_batch = topk_batch_opt(user_embeddings, item_embeddings)
    topk_batch = topk_batch.tolist()
    for cid, topk_list, past in zip(
            cid_batch, topk_batch, seq_items_batch):
        precision, ap = metrics(topk_list, past)
        precisions.append(precision)
        aps.append(ap)
        topk = ['0' + str(mapping[e]) for e in topk_list]
        predictions.write(cid + ',' + ' '.join(topk) + '\n')
    return sum(precisions)/len(precisions), sum(aps)/len(aps)


start = time.time()
data = jnp.load(_DATASET + '/tensors_history.npz')
if args.use_duplicate_features:
    items = data['items']
    seq_lengths = data['seq_lengths']
else:
    items = data['items_dedup']
    seq_lengths = data['seq_length_dedup']
item_embeddings = jnp.load(_EMBEDDINGS)['item_embeddings']
user_embeddings = jnp.load(_EMBEDDINGS)['user_embeddings']
mapping = pd.read_csv(_DATASET + '/item_map.csv')
cid_map = pd.read_csv(_DATASET + '/cid_map.csv')
cid_map = cid_map['customer_id'].to_list()
all_customers = set(pd.read_csv(
    _DATASET + '/customers.csv')['customer_id'].to_list())
print(f'Loaded from disk in {time.time() - start} secs.')
mapping.set_index('enum', inplace=True)
mapping = mapping['article_id'].to_dict()
print(user_embeddings.shape)
print(item_embeddings.shape)
print(items.shape)
print(len(mapping.keys()))
pbar = tqdm(cid_map)
pbar.set_description('KNN Search')
item_freq = defaultdict(lambda: 0)
sum_precision = None
sum_ap = None
count_precision = 0
with open(_DATASET + '/predictions.csv', 'w') as predictions:
    predictions.write('customer_id,prediction\n')
    seq_items_batch = []
    seq_lengths_batch = []
    cid_batch = []
    for index, cid in enumerate(pbar):
        item_history = items[index][:seq_lengths[index]]
        # Note: repeats ok.
        item_history = item_history[-_HISTORY:]
        for i in item_history:
            item_freq[i] += 1
        seq_items_batch.append(item_history)
        seq_lengths_batch.append(item_history.shape[0])
        cid_batch.append(cid)
        if len(seq_items_batch) == _BATCH:
            precision, ap = process_batch(
                user_embeddings[index:index + _BATCH],
                item_embeddings,
                seq_items_batch,
                seq_lengths_batch, cid_batch, predictions)
            if sum_precision is None:
                sum_precision = precision
                sum_ap = ap
            else:
                sum_precision += precision
                sum_ap += ap
            count_precision += 1
            pbar.set_description(
                f'KNN Search precision={sum_precision/count_precision:.4f} '
                f'map = {sum_ap/count_precision:.4f}')
            seq_lengths_batch = []
            seq_items_batch = []
            cid_batch = []
    if len(cid_batch) > 0:
        precision, ap = process_batch(
            user_embeddings[-len(cid_batch):],
            item_embeddings, seq_items_batch,
            seq_lengths_batch, cid_batch, predictions)
        if sum_precision is None:
            sum_precision = precision
            sum_ap = ap
        else:
            sum_precision += precision
            sum_ap += ap
        count_precision += 1
    print(f'Avg Precision = {sum_precision/count_precision:.4f} '
          f'map = {sum_ap/count_precision:.4f}')
    missing_customers = all_customers.difference(cid_map)
    global_top_k = list(item_freq.items())
    global_top_k.sort(key=lambda e: e[1], reverse=True)
    global_top_k = global_top_k[:_K]
    global_top_k = ['0' + str(mapping[e[0]]) for e in global_top_k]
    pbar = tqdm(missing_customers)
    pbar.set_description('Missing customers')
    for cid in pbar:
        predictions.write(cid + ',' + ' '.join(global_top_k) + '\n')
