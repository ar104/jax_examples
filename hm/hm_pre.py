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
    if abs(col_min - col_max) < 1e-6:
        df[col_name] = 0.5
    else:
        df[col_name] = (
            df[col_name].fillna(col_mean) - col_min)/(col_max - col_min)
    print(f'Encoded column {col_name} between '
          f'minimum {col_min} and maximum {col_max}')


def encode_customer_features(df_customers):
    encode_numerical(df_customers, 'age')
    encode_numerical(df_customers, 'FN')
    encode_numerical(df_customers, 'Active')
    df_customers['club_member_status'].fillna('UNK', inplace=True)
    encode_categorical(df_customers, 'club_member_status')
    df_customers['fashion_news_frequency'].fillna('UNK', inplace=True)
    encode_categorical(df_customers, 'fashion_news_frequency')
    encode_categorical(df_customers, 'postal_code')


def encode_article_features(df_articles):
    encode_categorical(df_articles, 'colour_group_name')
    encode_categorical(df_articles, 'section_name')
    encode_categorical(df_articles, 'garment_group_name')
    # enum to map transaction set to articles.
    df_articles['enum'] = list(range(df_articles.shape[0]))


start = time.time()
df_customers = pd.read_csv(_DATASET + '/customers.csv')
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
start_of_dataset = datetime.strptime(transactions['t_dat'].min(), '%Y-%M-%d')
transactions = pd.DataFrame(
    data={'customer_id': transactions['customer_id'].to_list(),
          'feature': list(zip(transactions['article_id'].to_list(),
                              [(datetime.strptime(e, '%Y-%M-%d') -
                                start_of_dataset).days
                               for e in transactions['t_dat'].to_list()]))})
print(f'Loaded data and encoded entity features in '
      f'{time.time() - start} seconds.')


tqdm.pandas(desc='Generate purchase lists.')
grouped_transactions = transactions.groupby(
    'customer_id')['feature'].progress_apply(lambda s: s.tolist())
training_features = grouped_transactions.to_dict()
del grouped_transactions
del transactions
training_examples_items = []
training_examples_times = []
seq_length = []
training_examples_customer = []
customer_age = []
customer_fn = []
customer_active = []
customer_club_member_status = []
customer_fashion_news_frequency = []
customer_postal_code = []


def parse_time(s):
    return datetime.strptime(s, '%Y-%m-%d')


pbar = tqdm(training_features.keys())
pbar.set_description('Generate Samples')
for k in pbar:
    training_examples_customer.append(k)
    customer_age.append(df_customers.loc[k]['age'])
    customer_fn.append(df_customers.loc[k]['FN'])
    customer_active.append(df_customers.loc[k]['Active'])
    customer_club_member_status.append(
        df_customers.loc[k]['club_member_status']
    )
    customer_fashion_news_frequency.append(
        df_customers.loc[k]['fashion_news_frequency']
    )
    customer_postal_code.append(
        df_customers.loc[k]['postal_code']
    )
    purchases = training_features[k]
    purchases.sort(key=lambda e: e[1])
    purchase_times = [e[1] for e in purchases[-HISTORY:]]
    purchases = [df_articles['enum'].loc[e[0]] for e in purchases[-HISTORY:]]
    seq_length.append(len(purchases))
    short = HISTORY - len(purchases)
    if short > 0:
        purchases.extend([-1]*short)
        purchase_times.extend([-1]*short)
    training_examples_items.append(purchases)
    training_examples_times.append(purchase_times)


del training_features

items = np.array(training_examples_items)
timestamps = np.array(training_examples_times)
seq_lengths = np.array(seq_length)
customer_age = np.array(customer_age)
customer_fn = np.array(customer_fn)
customer_active = np.array(customer_active)
customer_club_member_status = np.array(customer_club_member_status)
customer_fashion_news_frequency = np.array(customer_fashion_news_frequency)
customer_postal_code = np.array(customer_postal_code)
articles_colour_group_name = np.array(df_articles['colour_group_name'])
articles_section_name = np.array(df_articles['section_name'])
articles_garment_group_name = np.array(df_articles['garment_group_name'])
np.savez(_DATASET + '/tensors_history.npz',
         items=items,
         timestamps=timestamps,
         seq_lengths=seq_lengths,
         customer_age=customer_age,
         customer_fn=customer_fn,
         customer_active=customer_active,
         customer_club_member_status=customer_club_member_status,
         customer_fashion_news_frequency=customer_fashion_news_frequency,
         customer_postal_code=customer_postal_code,
         articles_color_group_name=articles_colour_group_name,
         articles_section_name=articles_section_name,
         articles_garment_group_name=articles_garment_group_name)
training_examples_customer = pd.DataFrame(
    {'customer_id': training_examples_customer})
training_examples_customer.to_csv(_DATASET + '/cid_map.csv', index=False)
print(f'items {items.shape} seq_lengths {seq_lengths.shape}')
