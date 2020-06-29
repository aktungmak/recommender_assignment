from datetime import datetime
from lightfm import LightFM
from scipy.sparse import coo_matrix
import numpy as np
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

def make_interaction_matrix(events_df):
    data = np.ones(events_df.visitorid.size)
    return coo_matrix((data, (events_df.visitorid, events_df.itemid)),
                      shape=(events_df.visitorid.size, events_df.itemid.size))

def make_model(interaction_matrix):
    # use WARP since only positive interactions are present
    model = LightFM(loss='warp')
    model.fit(interaction_matrix, epochs=1, num_threads=2)
    return model

def recommendations_for_user(model, userid, itemidsi, count=100):
    recommendations = model.predict(userid, itemids)
    recommendations[::-1].sort()
    return recommendations[:count]

if __name__ == '__main__':
    edf = clean_events('source_data/events.csv')
    im  = make_interaction_matrix(edf)
    model = make_model(im)

