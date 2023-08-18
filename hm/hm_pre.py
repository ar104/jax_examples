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
transactions = pd.read_csv(_DATASET + '\\transactions_train.csv')
unique_items = transactions['article_id'].unique()
df_unique = pd.DataFrame(
    {'article_id': unique_items, 'enum': list(range(len(unique_items)))})
df_unique.to_csv(_DATASET + '\\item_map.csv', index=False)
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


def parse_time(s):
    return datetime.strptime(s, '%Y-%m-%d')


pbar = tqdm(training_features.keys())
pbar.set_description('Generate Samples')
for k in pbar:
    purchases = training_features[k]
    purchases.sort(key=lambda e: e[1])
    history = [item_mapping[e[0]] for e in purchases[-HISTORY:]]
    seq_length.append(len(history))
    short = HISTORY - len(history)
    if short > 0:
        history.extend([-1]*short)
    training_examples_items.append(history)

del training_features
items = np.array(training_examples_items)
seq_lengths = np.array(seq_length)
np.savez(_DATASET + '//tensors_history.npz', items=items,
         seq_lengths=seq_lengths)
print(f'items {items.shape} seq_lengths {seq_lengths.shape}')
