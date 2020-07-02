from datetime import datetime
from lightfm import LightFM
from lightfm.evaluation import auc_score
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

def add_event_scoring(events_df):
    'convert event type to numerical value corresponding to event frequency'
    event_frequency = events_df.event.value_counts().to_dict()
    category_scores = {cat: 1 / (event_frequency[cat] / events_df.event.size)
                       for cat in events_df.event.dtype.categories}
    events_df['score'] = events_df.event.replace(category_scores)
    return events_df

def train_test_split(df, ratio=0.95):
    split_point = int(df.shape[0] * ratio)
    df_train    = df.iloc[0:split_point]
    df_test     = df.iloc[split_point:]

    df_test = df_test[(df_test['visitorid'].isin(df_train['visitorid'])) &
                      (df_test['itemid'].isin(df_train['itemid']))]

    return df_train, df_test

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

def make_model(interaction_matrix):
    # use WARP since only positive interactions are present
    model = LightFM(loss='warp')
    model.fit(interaction_matrix, epochs=10, num_threads=2)
    return model

def measure_model(model, test_im):
    test_auc  = auc_score(model, test_im, num_threads=4)
    return test_auc.mean()

def get_recommendation_visitorids(filename):
    df = pd.read_csv(filename)
    return df.visitorid

def make_recommendations(model, visitorids, itemids, count=100):
    recommendations = []
    for i, visitorid in enumerate(visitorids):
        scores = model.predict(visitorid, itemids)
        top_items = itemids[np.argsort(-scores)]
        recommendations.append([visitorid] + list(top_items[:count]))
        print(f'progress: {i} of {len(visitorids)}', end='\r')
    return recommendations

def save_recommendations_csv(recommendations, filename):
    df = pd.DataFrame(recommendations)
    headers = ['visitorid'] + [f'item_{i}' for i in range(0, len(df.columns) - 1)]
    df.columns = headers
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    edf         = clean_events('source_data/events.csv')
    edf         = add_event_scoring(edf)
    train, test = make_interaction_matrices(edf)
    model       = make_model(train)
  # mean_auc    = measure_model(model, test)
  # print('test AUC:', mean_auc)

    visitorids  = get_recommendation_visitorids('source_data/predictions.csv')
    itemids     = edf.itemid.unique()
    recs        = make_recommendations(model, visitorids, itemids)

    save_recommendations_csv(recs, 'restapi/predictions.csv')
