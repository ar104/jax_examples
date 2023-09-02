# Preprocess H&M data to generate labeled tensors.
from datetime import datetime
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import time


_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
HISTORY = 256

start = time.time()
transactions = pd.read_csv(_DATASET + '/transactions_train.csv')
unique_items = transactions['article_id'].unique()
df_unique = pd.DataFrame(
    {'article_id': unique_items, 'enum': list(range(len(unique_items)))})
df_unique.to_csv(_DATASET + '/item_map.csv', index=False)
df_unique.set_index('article_id', inplace=True)
item_mapping = df_unique['enum'].to_dict()
del df_unique

transactions = pd.DataFrame(
    data={'customer_id': transactions['customer_id'].to_list(),
          'feature': list(zip(transactions['article_id'].to_list(),
                              transactions['t_dat'].to_list()))})
print(f'Loaded transactions and computed mappings in '
      f'{time.time() - start} seconds.')


tqdm.pandas(desc='Generate purchase lists.')
grouped_transactions = transactions.groupby(
    'customer_id')['feature'].progress_apply(lambda s: s.tolist())
training_features = grouped_transactions.to_dict()
del grouped_transactions
del transactions
training_examples_items = []
seq_length = []
training_examples_items_dedup = []
seq_length_dedup = []
training_examples_customer = []


def parse_time(s):
    return datetime.strptime(s, '%Y-%m-%d')


pbar = tqdm(training_features.keys())
pbar.set_description('Generate Samples')
for k in pbar:
    training_examples_customer.append(k)
    purchases = training_features[k]
    purchases.sort(key=lambda e: e[1])
    purchases = [item_mapping[e[0]] for e in purchases[-HISTORY:]]
    deduped_purchases = list(set(purchases))
    # With dups
    seq_length.append(len(purchases))
    short = HISTORY - len(purchases)
    if short > 0:
        purchases.extend([-1]*short)
    training_examples_items.append(purchases)
    # Without dups
    seq_length_dedup.append(len(deduped_purchases))
    short = HISTORY - len(deduped_purchases)
    if short > 0:
        deduped_purchases.extend([-1]*short)
    training_examples_items_dedup.append(deduped_purchases)

del training_features

items = np.array(training_examples_items)
items_dedup = np.array(training_examples_items_dedup)
seq_lengths = np.array(seq_length)
seq_length_dedup = np.array(seq_length_dedup)

np.savez(_DATASET + '/tensors_history.npz',
         items=items,
         seq_lengths=seq_lengths,
         items_dedup=items_dedup,
         seq_length_dedup=seq_length_dedup)
training_examples_customer = pd.DataFrame(
    {'customer_id': training_examples_customer})
training_examples_customer.to_csv(_DATASET + '/cid_map.csv', index=False)
print(f'items {items.shape} seq_lengths {seq_lengths.shape}')
