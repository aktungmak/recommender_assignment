from datetime import datetime
from scipy.sparse import coo_matrix
import pandas as pd

def clean_events(input_file):
    df = pd.read_csv(input_file)
    # remove data from Sep 2015, as stated in the instructions
    sep1 = datetime.timestamp(datetime(2015, 9,  1)) * 1000
    oct1 = datetime.timestamp(datetime(2015, 10, 1)) * 1000
    df   = df[~df.timestamp.between(sep1, oct1)]
    # sort by timestamp
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    # convert event column to category type
    df.event = df.event.astype('category')
    # ignore transactionid for now
    return df.drop('transactionid', axis=1)

def extract_categories_from_properties(properties_file):
    df = pd.read_csv(properties_file)
    cats = df[df.property == 'categoryid']
    cats = cats.drop(['property'], axis=1)
    cats = df.rename(columns={'value': 'category'})
    return cats.sort_values(by='timestamp').reset_index(drop=True)

def make_interaction_matrices(events_df):
    train, test = train_test_split(events_df)

    n_visitors = events_df.visitorid.size
    n_items    = events_df.itemid.size

    train_im = coo_matrix((train.score, (train.visitorid, train.itemid)),
                          shape=(n_visitors, n_items))
    test_im  = coo_matrix((test.score, (test.visitorid, test.itemid)),
                          shape=(n_visitors, n_items))
    train_im.sum_duplicates()
    test_im.sum_duplicates()
    return train_im, test_im

def join_events_to_categories(event_df, category_df):
    '''join event and category data, using timestamp to find
       the category that applied at the time of the event'''
    df = event_df.merge(category_df, on='itemid', how='left',
                        suffixes=('', '_cat'))
    # not all itemids have categories, so drop them
    df = df[df.category.notna()]
    df['tstamp_diff'] = df.timestamp - df.timestamp_cat
    # drop category changes from the future
    df = df[df.tstamp_diff < 0]
    # drop all but the latest update
    df = df.drop_duplicates(subset=['timestamp', 'visitorid', 'event', 'itemid'],
                            keep='last')
    df = df.drop(['timestamp_cat', 'tstamp_diff'], axis=1)
    return df.reset_index(drop=True)

def make_model(interaction_matrix, item_features):
    model = LightFM(loss='warp')
    model.fit(interaction_matrix, item_features=item_features,
              epochs=10, num_threads=2)
    return model

if __name__ == '__main__':
    edf = clean_events('source_data/events.csv')
    cdf = extract_categories_from_properties('source_data/item_properties.csv')
    df  = join_events_to_categories(edf, cdf)
    train, test = make_interaction_matrices(df)
    model = make_model(train, df.category)
