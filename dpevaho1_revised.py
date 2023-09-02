#!/usr/bin/env python
# coding: utf-8

# # Module 3 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# ## k Nearest Neighbors and Model Evaluation
# 
# In this programming assignment you will use k Nearest Neighbors (kNN) to build a "model" that will estimate the compressive strength of various types of concrete. This assignment has several objectives:
# 
# 1. Implement the kNN algorithm with k=9. Remember...the data + distance function is the model in kNN. In addition to asserts that unit test your code, you should "test drive" the model, showing output that a non-technical person could interpret.
# 
# 2. You are going to compare the kNN model above against the baseline model described in the course notes (the mean of the training set's target variable). You should use 10 fold cross validation and Mean Squared Error (MSE):
# 
# $$MSE = \frac{1}{n}\sum^n_i (y_i - \hat{y}_i)^2$$
# 
# as the evaluation metric ("error"). Refer to the course notes for the format your output should take. Don't forget a discussion of the results.
# 
# 3. use validation curves to tune a *hyperparameter* of the model. 
# In this case, the hyperparameter is *k*, the number of neighbors. Don't forget a discussion of the results.
# 
# 4. evaluate the *generalization error* of the new model.
# Because you may have just created a new, better model, you need a sense of its generalization error, calculate that. Again, what would you like to see as output here? Refer to the course notes. Don't forget a discussion of the results. Did the new model do better than either model in Q2?
# 
# 5. pick one of the "Choose Your Own Adventure" options.
# 
# Refer to the "course notes" for this module for most of this assignment.
# Anytime you just need test/train split, use fold index 0 for the test set and the remainder as the training set.
# Discuss any results.

# ## Load the Data
# 
# The function `parse_data` loads the data from the specified file and returns a List of Lists. The outer List is the data set and each element (List) is a specific observation. Each value of an observation is for a particular measurement. This is what we mean by "tidy" data.
# 
# The function also returns the *shuffled* data because the data might have been collected in a particular order that *might* bias training.

# In[1]:


import random
from typing import List, Dict, Tuple, Callable


# In[2]:


def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


# In[3]:


data = parse_data("concrete_compressive_strength.csv")


# In[4]:


data[0]


# In[5]:


len(data)


# There are 1,030 observations and each observation has 8 measurements. The data dictionary for this data set tells us the definitions of the individual variables (columns/indices):
# 
# | Index | Variable | Definition |
# |-------|----------|------------|
# | 0     | cement   | kg in a cubic meter mixture |
# | 1     | slag     | kg in a cubic meter mixture |
# | 2     | ash      | kg in a cubic meter mixture |
# | 3     | water    | kg in a cubic meter mixture |
# | 4     | superplasticizer | kg in a cubic meter mixture |
# | 5     | coarse aggregate | kg in a cubic meter mixture |
# | 6     | fine aggregate | kg in a cubic meter mixture |
# | 7     | age | days |
# | 8     | concrete compressive strength | MPa |
# 
# The target ("y") variable is a Index 8, concrete compressive strength in (Mega?) [Pascals](https://en.wikipedia.org/wiki/Pascal_(unit)).

# ## Train/Test Splits - n folds
# 
# With n fold cross validation, we divide our data set into n subgroups called "folds" and then use those folds for training and testing. You pick n based on the size of your data set. If you have a small data set--100 observations--and you used n=10, each fold would only have 10 observations. That's probably too small. You want at least 30. At the other extreme, we generally don't use n > 10.
# 
# With 1,030 observations, n = 10 is fine so we will have 10 folds.
# `create_folds` will take a list (xs) and split it into `n` equal folds with each fold containing one-tenth of the observations.

# In[6]:


def create_folds(xs: List, n: int) -> List[List[List]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# In[7]:


folds = create_folds(data, 10)


# In[8]:


len(folds)


# We always use one of the n folds as a test set (and, sometimes, one of the folds as a *pruning* set but not for kNN), and the remaining folds as a training set.
# We need a function that'll take our n folds and return the train and test sets:

# In[9]:


def create_train_test(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test


# We can test the function to give us a train and test datasets where the test set is the fold at index 0:

# In[10]:


train, test = create_train_test(folds, 0)


# In[11]:


len(train)


# In[12]:


len(test)


# ## Answers
# 
# Answer the questions above in the space provided below, adding cells as you need to.
# Put everything in the helper functions and document them.
# Document everything (what you're doing and why).
# If you're not sure what format the output should take, refer to the course notes and what they do for that particular topic/algorithm.

# ## Problem 1: kNN
# 
# Implement k Nearest Neighbors with k = 9.

# In[13]:


from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
self_check = [
    [0.23, 0.81, 0.18],
    [0.42, 0.78, 0.33],
    [0.64, 0.23, 0.14],
    [0.87, 0.19, 0.17],
    [0.76, 0.43, 0.32]
]


# <a id="calculate_distance"></a>
# ## calculate_distance
# 
# Here we are using two lists being pulled in to calculate the distance using the Euclidean distance in order to build our models. This takes in the last value, or the y value, of the differing lines in the dataset and determines the distance between the y values. **Used by**: [k_nearest_neighbor](#k_nearest_neighbor).
# 
# * **train: List[List[float]]**: This value is a list of list of floats that contains the folds we are training on.
# * **test: List[List[float]]**: this is a list of list of floats that contains the fold we are using for validation.
# 
# **returns**: List[int, List[float]] The returned value is the distance from the point we are measuring against, then the measured values against the test set.

# In[14]:


def calculate_distance(train, test):
    return sqrt(sum([(xi_1 - xi_2)**2 for xi_1, xi_2 in zip(train, test)]))


# In[15]:


test1 = abs(calculate_distance(self_check[0][:-1], self_check[1][:-1]))
assert test1 == 0.19235384061671346
test2 = abs(calculate_distance(self_check[-1][:-1], self_check[1][:-1]))
assert test2 == 0.4879549159502341
test3 = abs(calculate_distance(self_check[-1][:-1], self_check[2][:-1]))
assert test3 == 0.233238075793812


# <a id="average_k_nearest_neighbors"></a>
# ## average_k_nearest_neighbors
# 
# This function has a simple purpose of taking the values from calculate distance, sorting them, then returning the top k performers in that list to be used to find the average. **Used by**: [k_nearest_neighbor](#k_nearest_neighbor).
# 
# * **neighbors: List[Tuple[float, List[float]]]**: This is an array that contains the distance that each of the points are from the test as well as the actual set that the distance was determined from as well.
# 
# **returns**: float: The value returned is the average of the k nearest neighbors that will be used to evaluate how accurate the model is.

# In[16]:


def average_k_nearest_neighbors(neighbors):
    return sum([value[1][-1] for value in neighbors])/len(neighbors)


# In[17]:


assert average_k_nearest_neighbors([(0, p) for p in self_check[1:]]) == 0.24
assert average_k_nearest_neighbors([(0, p) for p in self_check[2:]]) == 0.21000000000000005
assert average_k_nearest_neighbors([(0, p) for p in self_check[0:]]) == 0.22800000000000004


# <a id="k_nearest_neighbor"></a>
# ## k_nearest_neighbor
# 
# This function "builds a model," though it is technically a lazy algorithm, that will take in a  dataset and a query to determine the average of the k closest neighbors, which in turn is used to predict the values.  **Uses**: [calculate_distance](#calculate_distance), [average_k_nearest_neighbors](#average_k_nearest_neighbors) **Used by**: [learning_curve_creation](#learning_curve_creation).
# 
# * **calculate: function**: This is the function that is used to determine how y hat is being predicted, in the case in this assignment it is the euclidean distance.
# * **processing: function**: This function is used to determine what is the closest value, and in the case of this assignment that is the average of k nearest neighbors.
# * **k: int**: This value is simply for determining how many of the closest neighbors to look into.
# * **dataset: List[List[float]]**: here is the set of data that will be tested against
# 
# **returns**: function: float: this function returns the function built within that will return the average of the query being tested. 

# In[18]:


def k_nearest_neighbor(calculate, processing, k: int, dataset: List[List[float]]):
    def knn(query):
        distances = []
        for example in dataset:
            distances.append((calculate(example, query), example))
        distances.sort(key=lambda x: x[0])
        nearest = distances[0:k]
        return processing(nearest)
    return knn


# In[19]:


test1 = k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 2, self_check[:1])
assert test1(self_check[0]) == 0.18
test1 = k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 2, self_check[:3])
assert test1(self_check[0]) == 0.255
test1 = k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 2, self_check[:5])
assert test1(self_check[4]) == 0.23


# In[20]:


for i in range(0, 10):
    knn_model = k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 3, train[i:])
    print(train[i], knn_model(train[i][:-1]))


# ## Problem 2: Evaluation vs. The Mean
# 
# Using Mean Squared Error (MSE) as your evaluation metric, evaluate your implement above and the Null model, the mean.

# <a id="find_mean"></a>
# ## find_mean
# 
# The goal here is to take in the dataset, and find the mean. No matter what is asked, it will always return the mean of the whole dataset **Used by**: [cross_validation](#cross_validation)
# 
# * **dataset: List[List[float]]**: here is the set of data that will be tested against
# 
# **returns**: int: the mean of the data provided.

# In[21]:


def find_mean(dataset):
    mean = np.mean([y[-1] for y in dataset])
    def mean_baseline(query):
        return mean
    return mean_baseline


# In[22]:


test1 = find_mean(train[0:])
assert test1(test) < 36
test2 = find_mean(self_check[0:])
assert test2(self_check[0]) < 0.23
test3 = find_mean(self_check[1:])
assert test3(self_check[0]) == 0.24


# <a id="mean_squared_error"></a>
# ## mean_squared_error
# 
# This function is simply finding the mean squared error by summing up all of the differences between y and y_hat then dividing it by the number of values **Used by**: [cross_validation](#cross_validation)
# 
# * **ys: List[float]**: This contains all of the true values we are looking to guess in the dataset
# * **y_hats: List[float]**: this contains the predictions used in order to calculate the error percentage
# 
# **returns**: float: this is the error rate

# In[23]:


def mean_squared_error(ys: List[float], y_hats: List[float]):
    return sum([(y - y_hat)**2 for y, y_hat in zip(ys, y_hats)])/len(ys)


# In[24]:


testy, testyhat = [1, 2, 3, 4], [2, 3, 2, 3]
assert mean_squared_error(testy, testyhat) == 1
testy, testyhat = [5, 4, 1, 4], [10, 3, 2, 3]
assert mean_squared_error(testy, testyhat) == 7
testy, testyhat = [15, 14, 11, 4], [10, 3, 2, 3]
assert mean_squared_error(testy, testyhat) == 57


# <a id="evaluate_model"></a>
# ## evaluate_model
# 
# The purpose of this function is to use 10 crossfold validation by taking in the data as a whole, and also the folds created by create_folds, to have a test set created out of one of the 10 folds that is used to test the other 9, which then cycles down until each of the folds is used to validate the other data **Used by**: [cross_validation](#cross_validation) 
# 
# * **dataset: List[List[float]]**: here is the set of data that will be tested against
# * **model: function: float** This is the baseline model that will be tested against to determine error.
# * **evaluation_fn: function: float**: this is the function that is used to test how the model is doing vs the answers, for here it is mean squared error.
# 
#   
# **returns**: float: this function returns the error calculation of the test set and model, the ys and y_hats

# In[25]:


def evaluate_model(dataset, model, evaluation_fn):
    y_hats = []
    ys = []
    for t in dataset:
        ys.append(t[-1])
        prediction = model(t[:-1])
        y_hats.append(prediction)
    metric = evaluation_fn(ys, y_hats)
    return metric


# In[26]:


test1 = evaluate_model(self_check, find_mean(self_check), mean_squared_error) 
assert evaluate_model(self_check, find_mean(self_check), mean_squared_error) < 0.0065
test1 = evaluate_model(self_check[1:], find_mean(self_check[1:]), mean_squared_error) 
assert evaluate_model(self_check[1:], find_mean(self_check[1:]), mean_squared_error) == 0.00735
test1 = evaluate_model(self_check[2:], find_mean(self_check[2:]), mean_squared_error) 
assert evaluate_model(self_check[2:], find_mean(self_check[2:]), mean_squared_error)  == 0.0062


# <a id="cross_validation"></a>
# ## cross_validation
# 
# The purpose of this function is to use 10 crossfold validation by taking in the data as a whole, and also the folds created by create_folds, to have a test set created out of one of the 10 folds that is used to test the other 9, which then cycles down until each of the folds is used to validate the other data **Uses**: [mean_squared_error](#mean_squared), [evaluate_model](#evaluate_model) 
# 
# * **folds: List[List[List[float]]]**: here is the set of data that will be tested against, broken into 10 folds
# * **model_fn: function: float** This is the baseline model that will be tested against to determine error.
# * **evaluation_fn: function: float**: this is the function that is used to test how the model is doing vs the answers, for here it is mean squared error.
# 
#   
# **returns**: float: this function returns the error calculation of the test set and model, the ys and y_hats

# In[27]:


def cross_validation(folds, model_fn, evaluation_fn): 
    train_metrics, test_metrics, n = [], [], len(folds)
    for i in range(0, n):
        train, test = create_train_test(folds, i)
        model = model_fn(train)
        metric = evaluate_model(train, model, evaluation_fn)
        train_metrics.append(metric)
        metric = evaluate_model(test, model, evaluation_fn)
        test_metrics.append(metric)
    return train_metrics, test_metrics
    


# In[28]:


unit_folds = create_folds(self_check, 5)
unit_baseline_train1, unit_baseline_test1 = cross_validation(unit_folds, find_mean, mean_squared_error)
assert unit_baseline_train1 == [0.00735, 0.00481875, 0.0056500000000000005, 0.00701875, 0.005425]
assert unit_baseline_test1 ==  [0.0036, 0.01625625, 0.012099999999999998, 0.005256249999999997, 0.013224999999999997]
unit_baseline_train2, unit_baseline_test2 = cross_validation(unit_folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 9, self_check), mean_squared_error) 
assert unit_baseline_train2 == [0.007493999999999999, 0.005469000000000002, 0.006134000000000002, 0.007228999999999999, 0.005953999999999999]
assert unit_baseline_test2 == [0.002304000000000004, 0.010403999999999997, 0.007743999999999994, 0.003364000000000003, 0.008464000000000005]
unit_baseline_train3, unit_baseline_test3 = cross_validation(unit_folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 9, self_check[2:]), mean_squared_error)
assert unit_baseline_train3 == [0.008250000000000002, 0.004875000000000001, 0.007250000000000003, 0.008075000000000002, 0.005450000000000002]
assert unit_baseline_test3 == [0.0009, 0.014400000000000005, 0.004899999999999997, 0.001600000000000003, 0.012100000000000003]


# <a id="analyzation"></a>
# ## analyzation
# 
# This function simply takes in the test and train metrics then lists them out to have an easily read format of what is going on
# 
# * **train_metrics: List[List[float]]**: Here is the metrics from running the training set
# * **test_metrics: List[List[float]]**: Here is the metrics from running the testing set
# 
#   
# **returns**: float: this function returns the error calculation of the test set and model, the ys and y_hats

# In[29]:


def analyzation(train_metrics, test_metrics):
    train_avg, train_std, test_avg, test_std = np.mean(train_metrics), np.std(train_metrics), np.mean(test_metrics), np.std(test_metrics)
    test_avg = np.mean(test_metrics)
    print(f"Fold\tTrain\tTest")
    for i in range(len(train_metrics)):
        print(f"{i+1}: \t{train_metrics[i]:.2f}\t{test_metrics[i]:.2f}")
    print("\n\n")
    print(f"avg:\t{train_avg:.2f}\t{test_avg:.2f}")
    print(f"std:\t{train_std:.2f}\t{test_std:.2f}")


# In[30]:


unit_baseline_train1, unit_baseline_test1 = cross_validation(unit_folds, find_mean, mean_squared_error)
analyzation(unit_baseline_train1, unit_baseline_test1)
unit_baseline_train2, unit_baseline_test2 = cross_validation(unit_folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 9, self_check), mean_squared_error) 
analyzation(unit_baseline_train2, unit_baseline_test2)
unit_baseline_train3, unit_baseline_test3 = cross_validation(unit_folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 9, self_check[2:]), mean_squared_error)
analyzation(unit_baseline_train3, unit_baseline_test3)


# Asserts do not make sense in this function since the only thing it is doing is printing out the results of cross_validation. However, in order to test, three unit tests were provided to list the answers based on the self_check

# In[31]:


baseline_train, baseline_test = cross_validation(folds, find_mean, mean_squared_error)
knn_train, knn_test = cross_validation(folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 9, data), mean_squared_error)
analyzation(baseline_train, baseline_test)
analyzation(knn_train, knn_test)


# So one thing I noticed here was that even though the test set had a much higher standard deviation, the average was nearly exactly the same. Doing the math manually it was within decimal points. I looked at this for a long time trying to find if there was an error within the code but I could not find any errors, and this result was consistently happening. I am curious if there is something wrong somewhere down the line of code, but I could not find it, and even plugging the math in it was extremely close. 

# ## Problem 3: Hyperparameter Tuning
# 
# Tune the value of k.

# In[32]:


train, test = create_train_test(folds, 0)
train_ks = []
test_ks = []
for k in range(1, 21):
    _knn =  k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, k, train)
    metric = evaluate_model(train, _knn, mean_squared_error)
    train_ks.append(metric)
    metric = evaluate_model(test, _knn, mean_squared_error)
    test_ks.append(metric)
ks = list(range(1, 21))
plt.plot(ks, train_ks, 'b', ks, test_ks, 'r')
plt.xlabel('k')
plt.ylabel('MSE')
plt.gca().invert_xaxis()
plt.show();


# This was interesting to see the line change as the ks changed. It was odd to see how much ic could change but then once it started getting lower, the MSE really shrunk as well, which was expected in the train set and cool to see in the test set. It does look like the best k would be under 5, as it looks like it does continue a downward trend. Though the bowl is not being seen here either, which I thought was odd since sometimes the closest k can really skew data.

# ## Problem 4: Generalization Error
# 
# Analyze and discuss the generalization error of your model with the value of k from Problem 3.

# In[33]:


knn_train, knn_test = cross_validation(folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 3, data), mean_squared_error)
analyzation(knn_train, knn_test)


# Again, we see the averages being incredibly close, nearly identical, with a much different standard deviation. Though with k as equal to 3, that standard deviation did dramatically shrink. Which agian makes me wonder what is going on with the average. As far as I can tell it all should be calculating correctly, but something is missing.

# In[34]:


train_ks = []
test_ks = []
for k in range(1, 21):
    knn_train_results, knn_test_results = cross_validation(folds, lambda dataset: k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, k, data), mean_squared_error)
    train_ks.append(np.mean(knn_train_results))
    test_ks.append(np.mean(knn_test_results))
ks = list(range(1, 21))
plt.plot(ks, train_ks, 'b', ks, test_ks, 'r')
plt.xlabel('k')
plt.ylabel('MSE')
plt.gca().invert_xaxis()
plt.show();


# Here is a percect example of all of the means matching up. I do not understand what I did wrong to be getting the wrong, as it looks like the evalution_model is working correctly, but when cross_validation calls it, somehow the means keep ending up the exact same, even though the standard deviations are much different. Since this part was not required I did not troubleshoot it a ton due to time constraints, anything that could point out what was going on here would be appreciated though, since all my cross_validation showed the same results.

# ## Q5: Choose your own adventure
# 
# You have three options for the next part:
# 
# 1. You can implement mean normalization (also called "z-score standardization") of the *features*; do not normalize the target, y. See if this improves the generalization error of your model (middle).
# 
# 2. You can implement *learning curves* to see if more data would likely improve your model (easiest).
# 
# 3. You can implement *weighted* kNN and use the real valued GA to choose the weights. weighted kNN assigns a weight to each item in the Euclidean distance calculation. For two points, j and k:
# $$\sqrt{\sum w_i (x^k_i - x^j_i)^2}$$
# 
# You can think of normal Euclidean distance as the case where $w_i = 1$ for all features  (ambitious, but fun...you need to start EARLY because it takes a really long time to run).
# 
# The easier the adventure the more correct it must be...

# In[35]:


train_results, train_graph = [], []
test_results, test_graph = [], []
test_folds = create_folds(data, 20)
for i in range(0, len(test_folds)):
    train, test = create_train_test(test_folds, i)
    _knn =  k_nearest_neighbor(calculate_distance, average_k_nearest_neighbors, 3, train)
    metrics = evaluate_model(train, _knn, mean_squared_error)
    train_results.append(metrics)
    train_graph.append(sum(train_results)/len(train_results))
    metrics = evaluate_model(test, _knn, mean_squared_error)
    test_results.append(metrics)
    test_graph.append(sum(test_results)/len(test_results))
train_results.sort
test_results.sort
plt.plot(list(range(1, len(train_results)+1)), train_graph, 'b', list(range(1, len(test_results)+1)), test_graph, 'r')
plt.xlabel('fold')
plt.ylabel('MSE')
plt.show();


# Based of off the trend of this data, it looks like as it goes through, it has reached a spot where there does not appear to be any gain from having more data, however, it does look like there is a high variance, which I believe was expected. I am also concnered that what is affecting my data before is doing the same thing here.
# 
# I also created more folds here just to see what would happen, but it does look like that after 10 folds, the data stagnates and nothing more is learned from splitting the folds, which would make sense, because no more data is being added, just seperated more

# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.
