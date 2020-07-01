Case Study

Introduction

This is the first time I have tackled such a problem, and so I approached it in three stages, each will increasing levels of complexity.

The first iteration (model/model_1.py) took a naive approach.

In the second iteration (model/model_2.py), I chose to make use of the lightfm python package which is specialised for recommnedation tasks. It was the first time I had used this library so I kept things simple by only using the events dataset.

In the third iteration, I tried to incorporate the item category attribute, with the goal of making the predictions more accurate.

To deploy the model and make the predictions available for querying, I used the "bottle" python library to create a simple REST API (restapi/api.py). This was packaged into a docker container along with the CSV containing all the predictions and hosted using DigitalOcean since I already have a personal account there and use it for other projects.

Data Exploration

What data is present?
The core of the data is the events.csv file, which contains the details of the visitor interactions with the items between 2015-05-03 and 2015-09-18.

The table contains five columns:

timestamp: This is the time that the event occurred, presented as milliseconds since 1970-01-01. We might assume that more recent events are more representative of the visitor's preferences, since most people's behaviour changes over time. In the final model, perhaps we want to add a weighting to enforce this.

visitorid: This identifies the specific visitor that performed the action. It is also the key that will be used for generating recommendations in the final model.

event: This specifies the action taken by the user, one of view, addtocart and transactions. This is a categorical variable but intuitively a "transaction" event is more significant than a "view" event.
This is also shown by the data, we see that indeed there are

- 2 664 312 view events
- 69 332 addtocart events
- 22 457 transaction events

This seems natural since a user may view many items but they will only add a few to their basket and even fewer will actually purchase an item.
Therefore, we may consider weighting the event based on how common they are in the dataset, for example the following mapping could be used:

- view -> 1 / (n_view_events / n_events)
- addtocart -> 1 / (n_addtocart_events / n_events)
- transaction -> 1 / (n_transaction_events / n_events)

itemid: This identifies the specific item that the action was performed upon. The final model will produce recommendations of these items.

transactionid: This groups together items that were bought in the same transaction. This is quite a useful piece of data, since one can imagine that a visitor usually purchases items that are related to each other and so there may be a latent feature here.

To explore the data, the pandas library for python will be used. The clean_data.py script contains functions to:

- remove the events from September 2015, which the instructions said not to use (305924 events)
- Sort the events by timestamp to simplify later operations
- Convert the event column to a category type

The dataset is quite clean, the only cleanup needed was to order the events by timestamp to make some later operations easier. Otherwise, there was no other cleaning needed.

It was also discovered in the third iteration that many of the items are missing their category attribute. Since the dataset was reasonably large, it was chosen to simply drop the events with itemids lacking category attributes.

The model that will be created takes in one input visitor ID (one of the values from the visitorid column) and needs to produce as its output a 100-item list which represents the list of item IDs that the user would be most interested in based on the data that we have available, in order of descending interest.

It is assumed that there will be no new data coming in later, so no re-ranking is needed - the model can be run once over the entire dataset to produce all possible outcomes.

Iteration 1: Choice of algorithm and justification 
Since the base events.csv file only contains information on the users and their interactions with the items, this is purely a Collaborative recommendation task. In the third iteration where item properties are included, we can take a hybrid approach and also use Content-based techniques to potentially improve the results.

The first model attempt is to provide an aggregated score for each visitor's interaction with the items. We can consider a transaction event is rarer and more meaningful than a simple view event since users tend to view a lot of items but only buy those that they are actually interested in.
To quantify this we convert the events to a numerical value based on their frequency in the dataset. This means that a user buying an item is weighted higher than just viewing an item.

Additionally, we can consider that the user preference changes over time and so more recent events are more representative of the user's current interests. For this reason, the model weights recent events higher that earlier events using the timestamp.

These weightings are used to create an interaction matrix between visitorids and itemids.

Given this matrix, the model's task is to find the visitors that have the most similar item interactions to the given user, and then build the list of recommended items based on those that the nearest users interacted with but the user has not yet interacted with.

The measure of "the most similar item interactions" will be the dot product of the two user's feedback vectors. Dot product was chosen since the magnitude of the user vector is significant as well as its direction. This means that other similarity measures like the cosine are not appropriate since the only compare the direction of the user vectors.

Unfortunately this requires computing the dot product between each of the 1249704 users in the dataset which is very computationally expensive and took too long.

Iteration 2: Choice of algorithm and justification

# Which factors should be considered when evaluating the algorithms?
# Which algorithm is the best choice? Why?
# Which algorithms are bad choices? Why?

The next approach was to use the library lightfm to extract the embeddings from the interaction matrix and therefore reduce the computational load by not needing to calculate dot products between every user.

In this iteration, the scoring based on the relative frequency of the events described in iteration 1 was used again, but the weighting based on timestamp was skipped this time.

I chose the "warp" loss function since there are only positive interactions in the data and this loss function performs well under those conditions.

I left the hyperparameters at their defaults since I did not have enough knowledge to tune them meaningfully. Given more time, I would like to have used grid search in combination with cross-validation to identify which hyperparameters have a positive impact on the accuracy of the recommendations and also improve my own intuition.

Iteration 3: Choice of algorithm and justification

The goal of the third iteration was to utilise the "categoryid" attributes of the items. By joining the item_properties table with the events table on the 


Choice of metrics and justification 
Since I am new to the area of recommender systems, I chose to use the evaluation functions supplied with the lightfm library to measure the accuracy of the models in iteration 2 and 3. Iteration 1 is ignored in this section, since it was unsuccessful.

Specifically, I chose the ROC (receiver operating characteristic curve) as a measure of how well the model classifies a random recommendation against another. The ROC metric is the relation between the TPR (true positive rate) and FPR (false positive rate) which essentially compares how often the model guessed correctly against how often it guessed incorrectly.

The ROC is usually stated at a particular classification threshold. The AUC (area under the curve) on the other hand provides an aggregate measure of the ROC at all possible classification thresholds.

The lightfm library provides the lightfm.evaluation.auc_score function which is able to calculate the score for us given a particular model.

Unfortunately the performance of the AUC scoring functions was very poor, and it took an extremely long time to calculate the scores on the dataset (in excess of 2 hours). This meant that it was difficult to know exactly how 

From a personal reflection point of view, I would say that this is an area where I need to improve my knowledge most, since I feel that there are some optimisations to make the measurements quicker and more useful, and additionally I can investigate alternative measures of the model quality.

Easy to understand code and project structure 
There is very little code in this project, but to keep things simple I split the codebase into two parts: one that builds the model, and a second which is used to run the REST API.

Some of the code is duplicated between the 3 model iterations. If this was a real production project, we should only keep one of the models. Alternatively, the common code could be factored out to a separate module.

Hosting of API Endpoint

To host the API endpoint, I created a simple dockerfile that is based on the python:3.7 image and then adds the required dependencies via pip.

I then created a new droplet (compute instance) using my existing account at DigitalOcean and cloned the git repository containing my code to the instance. I could then use the following commands to start the container:

docker build Dockerfile
...

The benefit of this approach is that one can start the same docker container in any environment (e.g. a development machine) and have the same result as on the production server.

Bonus points for using item features 
Come back to this after everything else is done.
Not all events have corresponding properties, so either the events without properties must be dropped or dummy values added.
What features are there?
How can they be used?

Conclusion

Given an 80:20 train/test split of the data, the iteration 2 model achieved a mean AUC score of 0.9566047 on the test data.

Due to the restricted amount of time available for the task and also my limited experience with the tools in use, there are many possibilities that were not explored that could have increased the quality of the recommendations. For example:

- Making use of the transactionid column of the events table to understand which items were commonly bought together
- Using a more efficient approach for measuring the model, to decrease the cycle time required when testing the model
- Add unit testing for the code which cleans the data

This was an interesting learning experience for me,
