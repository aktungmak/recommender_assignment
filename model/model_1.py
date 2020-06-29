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
    '''create a feedback matrix from the events table, scoring
       the entries based on event type and timestamp.
       recent, rare event types score higher'''
    # convert event type to numerical value corresponding to frequency
    event_frequency = df.event.value_counts().to_dict()
    category_scores = {cat: 1 / (event_frequency[cat] / df.event.size)
                       for cat in df.event.dtype.categories}
    df['score'] = df.event.replace(category_scores)
    # use offset from first event to reduce magnitude of values
    df.timestamp = df.timestamp - df.timestamp.min()
    df['score'] = df.timestamp * df.score
    fm = coo_matrix((df.score, (df.visitorid, df.itemid)),
                    shape=(df.visitorid.size, df.itemid.size))
    fm.sum_duplicates()
    return fm

def create_recommendations_for_user(index, fm):
    pass

if __name__ == '__main__':
    edf = clean_events('source_data/events.csv')
    fm  = build_feedback_matrix(edf)

