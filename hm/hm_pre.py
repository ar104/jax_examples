# Preprocess H&M data to generate labeled tensors.
from datetime import datetime
import pandas as pd
import random
import numpy as np
from typing import Dict
from tqdm import tqdm
import time


_DATASET = 'C:\\Users\\aroym\\Downloads\\hm_data'
HISTORY = 256


def encode_categorical(df, col_name):
    vocab = df[col_name].unique()
    mapping = dict([(v, i) for i, v in enumerate(vocab)])
    df[col_name] = df[col_name].apply(lambda val: mapping[val])
    print(f'Encoded {len(mapping)} categories for column {col_name}')
    return mapping


def encode_numerical(df, col_name):
    col_min = df[col_name].min()
    col_max = df[col_name].max()
    col_mean = df[col_name].mean()
    df[col_name] = (
        df[col_name].fillna(col_mean) - col_min)/(col_max - col_min)
    print(f'Encoded column {col_name} between '
          f'minimum {col_min} and maximum {col_max}')


def encode_customer_features(df_customers):
    encode_numerical(df_customers, 'age')


def encode_article_features(df_articles):
    encode_categorical(df_articles, 'colour_group_name')
    encode_categorical(df_articles, 'section_name')
    encode_categorical(df_articles, 'garment_group_name')
    # enum to map transaction set to articles.
    df_articles['enum'] = list(range(df_articles.shape[0]))


start = time.time()
df_customers = pd.read_csv(_DATASET + '/customers.csv')[
    ['customer_id', 'age']]
df_articles = pd.read_csv(_DATASET + '/articles.csv')[
    ['article_id', 'colour_group_name', 'section_name', 'garment_group_name']]

encode_article_features(df_articles)
encode_customer_features(df_customers)

# Save mapping of articles for prediction step.
df_articles[['article_id', 'enum']].to_csv(_DATASET + '/item_map.csv')
# Setup indices for retrieval.
df_customers.set_index('customer_id', inplace=True)
df_articles.set_index('article_id', inplace=True)

# Read transaction data.
transactions = pd.read_csv(_DATASET + '/transactions_train.csv')
transactions = pd.DataFrame(
    data={'customer_id': transactions['customer_id'].to_list(),
          'feature': list(zip(transactions['article_id'].to_list(),
                              transactions['t_dat'].to_list()))})
print(f'Loaded data and encoded entity features in '
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
customer_age = []


def parse_time(s):
    return datetime.strptime(s, '%Y-%m-%d')


pbar = tqdm(training_features.keys())
pbar.set_description('Generate Samples')
for k in pbar:
    training_examples_customer.append(k)
    customer_age.append(df_customers.loc[k]['age'])
    purchases = training_features[k]
    purchases.sort(key=lambda e: e[1])
    purchases = [df_articles['enum'].loc[e[0]] for e in purchases[-HISTORY:]]
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
customer_age = np.array(customer_age)
articles_colour_group_name = np.array(df_articles['colour_group_name'])
articles_section_name = np.array(df_articles['section_name'])
articles_garment_group_name = np.array(df_articles['garment_group_name'])

np.savez(_DATASET + '/tensors_history.npz',
         items=items,
         seq_lengths=seq_lengths,
         items_dedup=items_dedup,
         seq_length_dedup=seq_length_dedup,
         customer_age=customer_age,
         articles_color_group_name=articles_colour_group_name,
         articles_section_name=articles_section_name,
         articles_garment_group_name=articles_garment_group_name)
training_examples_customer = pd.DataFrame(
    {'customer_id': training_examples_customer})
training_examples_customer.to_csv(_DATASET + '/cid_map.csv', index=False)
print(f'items {items.shape} seq_lengths {seq_lengths.shape}')
