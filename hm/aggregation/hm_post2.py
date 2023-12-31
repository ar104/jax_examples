import argparse
from collections import defaultdict
import pandas as pd
import jax
import jax.numpy as jnp
import pickle
import sys
from tqdm import tqdm
import time

sys.path.append('../common')   # noqa
from hm_model import _DIM   # noqa
from hm_aggregation import HMModel   # noqa
from hm_encoder import compute_pe_matrix   # noqa

_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
_EMBEDDINGS = _DATASET + '/embeddings_0.pickle'
_RETRIEVE_K = 200
_K = 12
_BATCH = 4096


@jax.jit
def topk_batch_opt(user_embeddings, item_embeddings):
    logits = jnp.einsum('ij,kj->ki', item_embeddings, user_embeddings)
    topk_logits, topk = jax.lax.top_k(logits, _RETRIEVE_K)
    return topk_logits, topk


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
        seq_items_batch,
        freq_batch,
        cid_batch,
        predictions):
    precisions = []
    aps = []
    topk_logits_batch, topk_batch = topk_batch_opt(
        user_embeddings, item_embeddings)
    topk_logits_batch = topk_logits_batch.tolist()
    topk_batch = topk_batch.tolist()
    for cid, topk_list, topk_logits, past, freq in zip(
            cid_batch, topk_batch, topk_logits_batch, seq_items_batch,
            freq_batch):
        topk_list = [(-freq[index], -logit, index)
                     for logit, index in zip(topk_logits, topk_list)]
        topk_list = [e[2] for e in sorted(topk_list)][:_K]
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
# Use duplicate items in history.
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
with open(f'{_EMBEDDINGS}', 'rb') as f:
    hm_model = pickle.load(f)
# Load and adjust item embeddings.
item_embeddings = hm_model.item_embedding_vectors(
    articles_color_group, articles_section_name, articles_garment_group)
mapping = pd.read_csv(_DATASET + '/item_map.csv')
cid_map = pd.read_csv(_DATASET + '/cid_map.csv')
cid_map = cid_map['customer_id'].to_list()
all_customers = set(pd.read_csv(
    _DATASET + '/customers.csv')['customer_id'].to_list())
print(f'Loaded from disk in {time.time() - start} secs.')
mapping.set_index('enum', inplace=True)
mapping = mapping['article_id'].to_dict()
print(items.shape)
print(len(mapping.keys()))
pbar = tqdm(cid_map)
pbar.set_description('KNN Search')
item_freq = defaultdict(lambda: 0)
sum_precision = None
sum_ap = None
count_precision = 0
pe_matrix = compute_pe_matrix(_DIM)
with open(_DATASET + '/predictions.csv', 'w') as predictions:
    predictions.write('customer_id,prediction\n')
    seq_items_batch = []
    freq_batch = []
    cid_batch = []
    customer_age_batch = []
    customer_fn_batch = []
    customer_active_batch = []
    customer_club_member_status_batch = []
    customer_fashion_news_frequency_batch = []
    customer_postal_code_batch = []
    customer_history_vector_batch = []
    for index, cid in enumerate(pbar):
        if len(seq_items_batch) == _BATCH:
            precision, ap = process_batch(
                hm_model.user_embedding_vectors(
                    jnp.stack(customer_history_vector_batch),
                    jnp.asarray(customer_age_batch),
                    jnp.asarray(customer_fn_batch),
                    jnp.asarray(customer_active_batch),
                    jnp.asarray(
                        customer_club_member_status_batch, jnp.int32),
                    jnp.asarray(
                        customer_fashion_news_frequency_batch, jnp.int32),
                    jnp.asarray(
                        customer_postal_code_batch, jnp.int32),
                ),
                item_embeddings,
                seq_items_batch,
                freq_batch,
                cid_batch,
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
            customer_fn_batch = []
            customer_active_batch = []
            customer_club_member_status_batch = []
            customer_fashion_news_frequency_batch = []
            customer_postal_code_batch = []
            customer_history_vector_batch = []
        item_history = items[index][:seq_lengths[index]]
        item_timestamps = timestamps[index][:seq_lengths[index]]
        for i in item_history:
            item_freq[i] += 1
        example_freq = defaultdict(lambda: 0)
        for i in items_dup[index][:seq_lengths_dup[index]]:
            example_freq[i] += 1
        seq_items_batch.append(item_history)
        customer_history_vector_batch.append(
            jnp.mean(
                hm_model.history_embedding_vectors(
                    item_embeddings[item_history, :] +
                    pe_matrix[item_timestamps, :]),
                axis=0))
        freq_batch.append(example_freq)
        cid_batch.append(cid)
        customer_age_batch.append(customer_age[index])
        customer_fn_batch.append(customer_fn[index])
        customer_active_batch.append(customer_active[index])
        customer_club_member_status_batch.append(
            customer_club_member_status[index])
        customer_fashion_news_frequency_batch.append(
            customer_fashion_news_frequency[index])
        customer_postal_code_batch.append(customer_postal_code[index])

    if len(cid_batch) > 0:
        precision, ap = process_batch(
            hm_model.user_embedding_vectors(
                jnp.stack(customer_history_vector_batch),
                jnp.asarray(customer_age_batch),
                jnp.asarray(customer_fn_batch),
                jnp.asarray(customer_active_batch),
                jnp.asarray(
                    customer_club_member_status_batch, jnp.int32),
                jnp.asarray(
                    customer_fashion_news_frequency_batch, jnp.int32),
                jnp.asarray(
                    customer_postal_code_batch, jnp.int32),
            ),
            item_embeddings,
            seq_items_batch, freq_batch, cid_batch, predictions)
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
