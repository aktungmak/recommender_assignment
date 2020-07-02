## Recommendation Engine Case Study

### Introduction

This is the first time I have tackled such a problem, and so I approached it in three stages, each with increasing levels of complexity.

In the first iteration I took a naive approach, based on my intuition of the problem and my thoughts of how it could be solved.

In the second iteration I chose to make use of the `lightfm` python package which is specialised for recommendation tasks. It was the first time I had used this library so I kept things simple by only using the events dataset.

In the third iteration, I tried to incorporate the item category attribute, with the goal of making the predictions more accurate.

To deploy the model and make the predictions available for querying, I used the `bottle` python library to create a simple REST API. This was packaged into a docker container along with the CSV containing all the predictions and hosted using DigitalOcean since I already have a personal account there and use it for other projects.

## Data Exploration

The core of the data is the `events.csv` file, which contains the details of the visitor interactions with the items (products) on the RetailRocket website between 2015-05-03 and 2015-09-18.

The table contains five columns:

__timestamp__: This is the time that the event occurred, presented as milliseconds since 1970-01-01. We might assume that more recent events are more representative of the visitor's preferences, since most people's behaviour changes over time. In the final model, perhaps we want to add a weighting to enforce this.

__visitorid__: This identifies the specific visitor that performed the action. It is also the key that will be used for generating recommendations in the final model.

__event__: This specifies the action taken by the user, one of view, addtocart and transactions. This is a categorical variable but intuitively a "transaction" event is more significant than a "view" event.
This is also shown by the data, we see that indeed there are

- 2 664 312 view events
- 69 332 addtocart events
- 22 457 transaction events

This seems natural since a user may view many items but they will only add a few to their basket and even fewer will actually purchase an item.
Therefore, we may consider weighting the event based on how common they are in the dataset, for example the following mapping could be used:

- view -> 1 / (n_view_events / n_events)
- addtocart -> 1 / (n_addtocart_events / n_events)
- transaction -> 1 / (n_transaction_events / n_events)

__itemid__: This identifies the specific item that the action was performed upon. The final model will produce recommendations of these items.

__transactionid__: This groups together items that were bought in the same transaction. This is quite a useful piece of data, since one can imagine that a visitor usually purchases items that are related to each other and so there may be a latent feature here.

To explore the data, the pandas library for python was used. Each of the models contain functions to:

- remove the events from September 2015, which the instructions said not to use (305924 events)
- Sort the events by timestamp to simplify later operations
- Convert the event column to a category type

The dataset is quite clean, the only cleanup needed was to order the events by timestamp to make some later operations easier. Otherwise, there was no other cleaning needed.

It was also discovered in the third iteration that many of the items are missing their category attribute. Since the dataset was reasonably large, it was chosen to simply drop the events with itemids lacking category attributes.

The model that will be created takes in one input visitor ID (one of the values from the visitorid column) and needs to produce as its output a 100-item list which represents the list of item IDs that the user would be most interested in based on the data that we have available, in order of descending interest.

It is assumed that there will be no new data coming in later, so no re-ranking is needed - the model can be run once over the entire dataset to produce all possible outcomes and this is what is deployed for the REST API to use.

## Iteration 1: Choice of algorithm and justification 

The code for this iteration can be found in the file `model/model_1.py`.

Since the base `events.csv` file only contains information on the users and their interactions with the items, this is purely a Collaborative recommendation task. In the third iteration where item properties are included, we can take a hybrid approach and also use Content-based techniques to potentially improve the results.

The first model attempt focuses on providing an aggregated score for each visitor's interactions with the items. We can consider a transaction event is rarer and more meaningful than a simple view event since users tend to view a lot of items but only buy those that they are actually interested in.
To quantify this we convert the events to a numerical value based on their frequency in the dataset. This means that a user buying an item is weighted higher than just viewing an item.

Additionally, we can consider that the user preference changes over time and so more recent events are more representative of the user's current interests. For this reason, the model weights recent events higher that earlier events using the timestamp.

These weightings are used to create an interaction matrix between visitorids and itemids, where each entry is the sum of all the interactions a particular user had with a particular item.

Given this matrix, the model's task is to find the visitors that have _the most similar item interactions_ to the given user, and then build the list of recommended items based on those that the nearest users interacted with but the user has not yet interacted with.

The measure of "the most similar item interactions" will be the dot product of the two user's feedback vectors. Dot product was chosen since the magnitude of the user vector is significant as well as its direction. This means that other similarity measures like the cosine are not appropriate since the only compare the direction of the user vectors.

Unfortunately this requires computing the dot product between each of the 1249704 users in the dataset which is very computationally expensive and took too long. For this reason, this approach was abandoned early on.

## Iteration 2: Choice of algorithm and justification

The code for this iteration is located in the file `model/model_2.py`.

The next approach was to use the library `lightfm` to extract the latent embeddings from the interaction matrix and therefore reduce the computational load by not needing to calculate dot products for every item between every user.

In this iteration, the scoring based on the relative frequency of the events described in iteration 1 was used again, but the weighting based on timestamp was skipped this time.

I chose the "warp" loss function since there are only positive interactions in the data and this loss function performs well under those conditions. Additionally, it focuses on optimising the results at the top of the recommendation list (Precision @ k) which is exactly the use case here, since we will only recommend the top 100 items.

I left the hyperparameters at their defaults since I did not have enough knowledge to tune them meaningfully. Given more time, I would like to have used grid search in combination with cross-validation to identify which hyperparameters have a positive impact on the accuracy of the recommendations and also improve my own intuition.

## Iteration 3: Choice of algorithm and justification

`model/model_3.py` contains the code for this iteration.

The goal of the third iteration was to utilise the "categoryid" attributes of the items. By joining the `item_properties` table with the events table on the `itemid` key I was able to provide the model with item features which has the potential to reveal stronger latent features of the items themselves. Intuitively, a user who interacted with an item in one category is clearly interested in that category and so has a higher probability of being interested in other items in the same category.

The problem with this approach was that not all items have corresponding categoryid properties, so either the events without properties must be dropped or dummy values added. In this case, I chose to simply drop the events without categoryids.

Other than the extraction and joining of the categoryid attributes, this is very similar to the model in iteration 2.

## Choice of metrics and justification 

Since I am new to the area of recommender systems, I chose to use the evaluation functions supplied with the lightfm library to measure the accuracy of the models in iteration 2 and 3. Iteration 1 is ignored in this section, since it was unsuccessful.

I selected two metrics for measuring the quality of the model: Precision @ k and ROC AUC.

Precision @ k is a measure of the proportion of the top k items which are positively rated (that is, the fraction of relevant recommendations).

ROC (receiver operating characteristic curve) is a measure of the probability that the model will rank a randomly chosen positive example higher than a randomly chosen negative example. The ROC metric is the relation between the TPR (true positive rate) and FPR (false positive rate) which essentially compares how often the model guessed correctly against how often it guessed incorrectly.

The ROC is usually stated at a particular classification threshold. The AUC (area under the curve) on the other hand provides an aggregate measure of the ROC at all possible classification thresholds.

The lightfm library provides the lightfm.evaluation.auc_score and lightfm.evaluation.precision_at_k function which is able to calculate the score for us given a particular model.

Unfortunately the performance of these scoring functions was very poor on this dataset, and it took an extremely long time to calculate the scores on the dataset (in excess of 6 hours). This meant that it was difficult to know exactly how changes in the hyperparameters were having an effect.

From a personal reflection point of view, I would say that this is an area where I need to improve my knowledge most, since I feel that there are some optimisations to make the measurements quicker and more useful, and additionally I can investigate alternative measures of the model quality.

## Code and project structure 

There is very little code in this project, but to keep things simple I split the codebase into two parts: one that builds the model, and a second which is used to run the REST API.

Some of the code is duplicated between the 3 model iterations. If this was a real production project, we should only keep one of the models. Alternatively, the common code could be factored out to a separate module.

## Hosting of API Endpoint

To host the API endpoint, I created a simple dockerfile that is based on the `python:3.7` image and then adds the required dependencies via pip.

I then created a new droplet (compute instance) using my existing account at DigitalOcean and cloned the git repository containing my code to the instance. I could then use the following commands to start the container:

```
docker build https://github.com/aktungmak/recommender_assignment.git#master:restapi -t aktungmak/recommender_assignment
docker run -d -p 8080:8080 aktungmak/recommender_assignment
```

The benefit of this approach is that one can start the same docker container in any environment (e.g. a development machine) and have the same result as on the production server.

## Conclusion

Given an 80:20 train/test split of the data, the iteration 2 model achieved a mean AUC score of 0.9566047 on the test data.

Due to the restricted amount of time available for the task and also my limited experience with the tools in use, there are many possibilities that were not explored that could have increased the quality of the recommendations. For example:

- Making use of the transactionid column of the events table to understand which items were commonly bought together
- Using a more efficient approach for measuring the model, to decrease the cycle time required when testing the model
- Tune the hyperparameters of the model using a grid search approach to help improve the AUC and Precision@K scores
- Add unit testing for the code, to ensure that the transformations being applied to the data were correct

This was an interesting learning experience for me, and I feel like I have a much better understanding of the tasks needed to set up a recommendation engine. What's more, I have a better view of the areas I have yet to understand so this will make it easier for me to learn faster in future.

