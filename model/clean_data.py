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

def build_feedback_matrix(df):
    # convert timestamp to offset since first event to reduce magnitude of values
    df.timestamp = df.timestamp - df.timestamp.min()
    # convert event type to numerical value corresponding to frequency
    event_frequency = df.event.value_counts().to_dict()
    category_scores = {cat: 1 / (event_frequency[cat] / df.event.size)
                       for cat in df.event.dtype.categories}
    df['score'] = df.event.replace(category_scores)
    # more recent events score higher
    df['score'] = df.timestamp * df.score
    fm = coo_matrix((df.score, (df.visitorid, df.itemid)),
                    shape=(df.visitorid.size, df.itemid.size))
    fm.sum_duplicates()
    return fm

def calculate_scores(index, fm):
    x = fm.getrow(index).toarray()
    result = []
    for i in range(1, len(fm.row)):
        y = fm.getrow(i).toarray().transpose()
        result.append((x.dot(y), i))

def make_category_file(input_file, output_file):
    df = pd.read_csv(input_file)
    cats = df[df.property == 'categoryid']
    cats = cats.sort_values(by='timestamp').reset_index(drop=True)
    cats.to_csv(output_file, index=False)

def join_event_category(event_df, category_df):
    '''join event and category data, using timestamp to find
       the category that applied at the time of the event'''
    df = event_df.merge(category_df, on='itemid', how='left',
                        suffixes=('', '_cat'))
    # not all itemids have categories, so drop them
    df = df[df.value.notna()]
    df['tstamp_diff'] = df.timestamp - df.timestamp_cat
    # drop category changes from the future
    df = df[df.tstamp_diff < 0]
    # drop all but the latest update
    df = df.drop_duplicates(subset=['timestamp', 'visitorid', 'event', 'itemid'],
                            keep='last')
    df = df.drop(['timestamp_cat', 'tstamp_diff'], axis=1)
    return df.reset_index(drop=True)

if __name__ == '__main__':
    edf = clean_events('source_data/events.csv')
    cdf = pd.read_csv('source_data/item_categories.csv')
    df  = join_event_category(edf, cdf)
    df  = df.drop(['property'], axis=1)
