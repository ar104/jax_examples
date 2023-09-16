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
_RETRIEVE_K = 200
_K = 12
_BATCH = 4096


@jax.jit
def topk_batch_opt(user_embeddings, item_embeddings):
    logits = jnp.einsum('ij,kj->ki', item_embeddings, user_embeddings)
    _, topk = jax.lax.top_k(logits, _RETRIEVE_K)
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


def process_batch(
        user_embeddings,
        item_embeddings,
        customer_age_vector,
        seq_items_batch,
        freq_batch,
        cid_batch,
        customer_age_batch,
        predictions):
    precisions = []
    aps = []
    customer_age_batch_array = jnp.asarray(customer_age_batch)
    user_embeddings = (
        user_embeddings +
        jnp.expand_dims(customer_age_batch_array, axis=1) *
        jnp.expand_dims(customer_age_vector, axis=0)
    )
    topk_batch = topk_batch_opt(user_embeddings, item_embeddings)
    topk_batch = topk_batch.tolist()
    for cid, topk_list, past, freq in zip(
            cid_batch, topk_batch, seq_items_batch, freq_batch):
        topk_list = [(-freq[index], index) for index in topk_list]
        topk_list = [e[1] for e in sorted(topk_list)][:_K]
        precision, ap = metrics(topk_list, past)
        precisions.append(precision)
        aps.append(ap)
        topk = ['0' + str(mapping[e]) for e in topk_list]
        predictions.write(cid + ',' + ' '.join(topk) + '\n')
    return sum(precisions)/len(precisions), sum(aps)/len(aps)


start = time.time()
data = jnp.load(_DATASET + '/tensors_history.npz')
items_dup = data['items']
seq_lengths_dup = data['seq_lengths']
items = data['items_dedup']
seq_lengths = data['seq_length_dedup']
customer_age = data['customer_age']
articles_color_group = data['articles_color_group_name']
articles_section_name = data['articles_section_name']
articles_garment_group = data['articles_garment_group_name']
saved_state = jnp.load(_EMBEDDINGS)
# Load and adjust item embeddings.
item_embeddings = saved_state['item_embeddings']
color_group_embeddings = saved_state['color_group_embeddings']
section_name_embeddings = saved_state['section_name_embeddings']
garment_group_embeddings = saved_state['garment_group_embeddings']
item_embeddings = (
    item_embeddings +
    color_group_embeddings[articles_color_group] +
    section_name_embeddings[articles_section_name] +
    garment_group_embeddings[articles_garment_group]
)
user_embeddings = saved_state['user_embeddings']
customer_age_vector = saved_state['customer_age_vector']
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
    freq_batch = []
    cid_batch = []
    customer_age_batch = []
    for index, cid in enumerate(pbar):
        if len(seq_items_batch) == _BATCH:
            precision, ap = process_batch(
                user_embeddings[index-_BATCH:index],
                item_embeddings,
                customer_age_vector,
                seq_items_batch,
                freq_batch,
                cid_batch,
                customer_age_batch,
                predictions)
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
            seq_items_batch = []
            freq_batch = []
            cid_batch = []
            customer_age_batch = []
        item_history = items[index][:seq_lengths[index]]
        item_history = item_history[-_HISTORY:]
        for i in item_history:
            item_freq[i] += 1
        example_freq = defaultdict(lambda: 0)
        for i in items_dup[index][:seq_lengths_dup[index]]:
            example_freq[i] += 1
        seq_items_batch.append(item_history)
        freq_batch.append(example_freq)
        cid_batch.append(cid)
        customer_age_batch.append(customer_age[index])

    if len(cid_batch) > 0:
        precision, ap = process_batch(
            user_embeddings[-len(cid_batch):],
            item_embeddings,
            customer_age_vector,
            seq_items_batch,
            freq_batch,
            cid_batch,
            customer_age_batch,
            predictions)
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
