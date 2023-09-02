#!/usr/bin/env python
# coding: utf-8

# # Module 9 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# ## Naive Bayes Classifier
# 
# For this assignment you will be implementing and evaluating a Naive Bayes Classifier with the same data from last week:
# 
# http://archive.ics.uci.edu/ml/datasets/Mushroom
# 
# (You should have downloaded it).
# 
# <div style="background: lemonchiffon; margin:20px; padding: 20px;">
#     <strong>Important</strong>
#     <p>
#         No Pandas. The only acceptable libraries in this class are those contained in the `environment.yml`. No OOP, either. You can used Dicts, NamedTuples, etc. as your abstract data type (ADT) for the the tree and nodes.
#     </p>
# </div>
# 
# 
# You'll first need to calculate all of the necessary probabilities using a `train` function. A flag will control whether or not you use "+1 Smoothing" or not. You'll then need to have a `classify` function that takes your probabilities, a List of instances (possibly a list of 1) and returns a List of Tuples. Each Tuple has the best class in the first position and a dict with a key for every possible class label and the associated *normalized* probability. For example, if we have given the `classify` function a list of 2 observations, we would get the following back:
# 
# ```
# [("e", {"e": 0.98, "p": 0.02}), ("p", {"e": 0.34, "p": 0.66})]
# ```
# 
# when calculating the error rate of your classifier, you should pick the class label with the highest probability; you can write a simple function that takes the Dict and returns that class label.
# 
# As a reminder, the Naive Bayes Classifier generates the *unnormalized* probabilities from the numerator of Bayes Rule:
# 
# $$P(C|A) \propto P(A|C)P(C)$$
# 
# where C is the class and A are the attributes (data). Since the normalizer of Bayes Rule is the *sum* of all possible numerators and you have to calculate them all, the normalizer is just the sum of the probabilities.
# 
# You will have the same basic functions as the last module's assignment and some of them can be reused or at least repurposed.
# 
# `train` takes training_data and returns a Naive Bayes Classifier (NBC) as a data structure. There are many options including namedtuples and just plain old nested dictionaries. **No OOP**.
# 
# ```
# def train(training_data, smoothing=True):
#    # returns the Decision Tree.
# ```
# 
# The `smoothing` value defaults to True. You should handle both cases.
# 
# `classify` takes a NBC produced from the function above and applies it to labeled data (like the test set) or unlabeled data (like some new data). (This is not the same `classify` as the pseudocode which classifies only one instance at a time; it can call it though).
# 
# ```
# def classify(nbc, observations, labeled=True):
#     # returns a list of tuples, the argmax and the raw data as per the pseudocode.
# ```
# 
# `evaluate` takes a data set with labels (like the training set or test set) and the classification result and calculates the classification error rate:
# 
# $$error\_rate=\frac{errors}{n}$$
# 
# Do not use anything else as evaluation metric or the submission will be deemed incomplete, ie, an "F". (Hint: accuracy rate is not the error rate!).
# 
# `cross_validate` takes the data and uses 10 fold cross validation (from Module 3!) to `train`, `classify`, and `evaluate`. **Remember to shuffle your data before you create your folds**. I leave the exact signature of `cross_validate` to you but you should write it so that you can use it with *any* `classify` function of the same form (using higher order functions and partial application). If you did so last time, you can reuse it for this assignment.
# 
# Following Module 3's discussion, `cross_validate` should print out the fold number and the evaluation metric (error rate) for each fold and then the average value (and the variance). What you are looking for here is a consistent evaluation metric cross the folds. You should print the error rates in terms of percents (ie, multiply the error rate by 100 and add "%" to the end).
# 
# To summarize...
# 
# Apply the Naive Bayes Classifier algorithm to the Mushroom data set using 10 fold cross validation and the error rate as the evaluation metric. You will do this *twice*. Once with smoothing=True and once with smoothing=False. You should follow up with a brief explanation for the similarities or differences in the results.

# In[1]:


import random
from typing import List, Dict, Tuple, Callable
import numpy as np
self_check = [
    ['n', 'r', 'l', 'b'],
    ['y', 's', 'l', 'g'],
    ['n', 's', 's', 'r'],
    ['y', 'r', 's', 'r'],
    ['n', 's', 's', 'b'],
    ['n', 'r', 's', 'b'],
    ['y', 'r', 's', 'r'],
    ['n', 's', 's', 'g'],
    ['y', 'r', 'l', 'g'],
    ['y', 's', 'l', 'g'],
    ['n', 's', 'l', 'r'],
    ['y', 's', 'l', 'g'],
    ['y', 'r', 'l', 'r'],
    ['n', 's', 's', 'r'],
    ['n', 'r', 's', 'g']
]


# <a id="parse_data"></a>
# ## parse_data
# 
# This function was the parsing function given to us in the third module, which reads in a file to turn the data into a useable list by taking in the filename 
# 
# 
# * **file_name**: str:  The name of the file in the local directory.
# 
# 
# **returns**: list[list[str]]: Returns a list of lists of the data in the file

# In[2]:


def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [value for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


# In[3]:


training_data = parse_data("agaricus-lepiota.data")


# <a id="create_folds"></a>
# ## create_folds
# 
# This function was given to use from module 3, and its intention is to break the data given into 10 separate folds for cross validations 
# 
# 
# * **xs**:  list: data that will be broken down into an equal portion.
# * **n**: int:  the amount of folds to be created.
# 
# **returns**:list[list[list]]: returns a list of lists of lists that includes the folds where the data is broken into.

# In[4]:


def create_folds(xs: List, n: int) -> List[List[List]]:
    k, m = divmod(len(xs), n)
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# In[5]:


folds = create_folds(training_data, 10)
unit_folds = create_folds(self_check, 5)


# <a id="create_train_test"></a>
# ## create_train_test
# 
# This function was given to use from module 3, and its intention is to break the data given into 9 training sets and 1 test set to work through the models 
# 
# 
# * **folds**:  list[list[list]]: data that will be broken down 9 training sets and 1 test set that was broken into folds.
# * **n**: int:  the amount of folds to be created.
# 
# **returns**:tuple[list[list], list[[list]]: returns a tuple of values that contains lists of lists broken down into the training and test set

# In[6]:


def create_train_test(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test


# In[7]:


training, testing = create_train_test(folds, 0)
unit_train, unit_test = create_train_test(unit_folds, 0)


# For unit tests for create_train_test, create_folds, and parse_data, I did not think that unit tests were something very applicable, one because these have been written in previous assignments, and two because of the random of the data and while I could do one unit test with the self_check I did not think it was needed.

# <a id="unique_attributes"></a>
# ## unique_attributes
# 
# This function looks at a list of data to determine how many unique values are in there to be tested and create a domain  **Used by**: [id3](#id3), [homogeneous](#homogeneous), [get_majority_label](#get_majority_label), [pick_best_attribute](#pick_best_attribute)
# 
# * **attributes**: list:  A list attributes to be used to determine how many unique values there are .
# 
# 
# **returns**: list: returns a list that are unique from the attributes provided

# In[8]:


def unique_attributes(attributes: list[str]):
    unique = []
    for a in attributes:
        if a not in unique:
            unique.append(a)
    return unique


# In[9]:


unit_test = unique_attributes(['e', 'p', 'e', 'p'])
assert unit_test == ['e', 'p']
unit_test = unique_attributes([1, 1, 2, 2, 3, 3, 1, 1, 1])
assert unit_test == [1, 2, 3]
unit_test = unique_attributes(['words', 1, 'words', 1])
assert unit_test == ['words', 1]


# <a id="domain_value_amount"></a>
# ## domain_value_amount
# 
# This function simply finds how many of a value is in a specific feature by recieving a column, a value, then looking into that specific column for the specific value **Used by**: [train](#train).
# 
# * **training_data**: list[list[str]]: This is the data pulled that will build the model.
# * **columns**: int: the column of data that is trying to be found.
# * **value**: str: the value that it is trying to be determine how many there is
# 
# **returns**: int: the count of how many of a specific value is in a column

# In[10]:


def domain_value_amount(training_data: list[list[str]], column: int, value: str):
    count = 0
    for row in training_data:
        if row[column] == value:
            count += 1
    return count


# In[11]:


unit_test = domain_value_amount(training_data, 0, 'e')
assert unit_test == 4208
unit_test = domain_value_amount(training_data, 0, 'p')
assert unit_test == 3916
unit_test = domain_value_amount(training_data, 1, 'x')
assert unit_test == 3656


# <a id="create_domains"></a>
# ## create_domains
# 
# This is a simple function that takes the training data, and makes the rows columns and vise versa for ease of use **Used by**: [train](#train).
# 
# * **training_data**: list[list[str]]: This is the data pulled that will build the model.
# 
# 
# **returns**: list[list[str]] : returns essentially a list of the data where the rows are now the columns for ease of use

# In[12]:


def create_domains(training_data: list[list[str]]):
    domains = []
    for column in list(range(len(training_data[0]))):
        domain_row = []
        for row in training_data:
            domain_row.append(row[column])
        domains.append(domain_row)
    return domains


# In[13]:


unit_test = create_domains([[1, 1,], [2, 2], [3, 3]])
assert unit_test[0] == [1, 2, 3]
unit_test = create_domains([['p', 'x', 'l'], ['a', 'a', 0], [2, 'e', 10]])
assert unit_test[0] == ['p', 'a', 2]
unit_test = create_domains([['this', 'x', 'l'], ['is', 'a', 0], ['column', 'e', 10]])
assert unit_test[1] == ['x', 'a', 'e']


# <a id="calculate_probability"></a>
# ## calculate_probability
# 
# The purpose of this function is to take in the training data, as well as specified columns and values that will be evaluated and determine for each class label what the probability of a class label for a specified feature value **Used by**: [train](#train).
# 
# * **training_data**: list[list[str]]: This is the data pulled that will build the model.
# * * **columns**: int: the column of data that is trying to be found.
# * **value**: str: the value that it is trying to be determine how many there is
# * **smoothing**: bool: this value determines if smoothing will be used or not.
# * **smoothing_amount**: int: if smoothing is used, this determines the value for the bottom addition
# 
# **returns**: documentation of the returned value and type.

# In[14]:


def calculate_probability(training_data: list[list[str]], column: int, value: str, class_labels: list[str], smoothing: bool, smoothing_amount: int):
    label_probabilities = {}
    for label in class_labels:
        total_class, total_value = 0, 0
        for row in training_data:
            if row[0] == label: total_class += 1
            if row[0] == label and row[column] == value: total_value += 1
        if total_class == total_value and smoothing == True:
            total = len(training_data) + smoothing_amount
            label_probabilities[label] = float((total_value + 1) / total)
        elif total_class == total_value:
            total = len(training_data)
            label_probabilities[label] = float(total_value / total)
        elif smoothing == True:
            label_probabilities[label] = float((total_value + 1) / (total_class + smoothing_amount))
        else:
            label_probabilities[label] = float(total_value / total_class)
    return label_probabilities


# In[15]:


unit_test = calculate_probability(self_check, 1, 'b', ['y', 'n'], True, 4)
assert unit_test == {'y': 0.09090909090909091, 'n': 0.08333333333333333}
unit_test = calculate_probability(self_check, 0, 'b', ['y', 'n'], True, 1)
assert unit_test == {'y': 0.125, 'n': 0.1111111111111111}
unit_test = calculate_probability(self_check, 1, 's', ['y', 'n'], True, 1)
assert unit_test == {'y': 0.5, 'n': 0.6666666666666666}


# <a id="train"></a>
# ## train
# 
# This function is used to train a model by creating a Naive Bayes Classifier given specific data to train it. There is an optional value to determine if smoothing will be used or not to see how it will change the data. Essentially it finds the percents of each value for each feature, as a probability, then using those probabilities in the model it will be used to determine what the class label should be later by looking at the probability specific features had a speciifc class label **Uses**: [unique_attributes](#unique_attributes), [create_domains](#create_domains), [calculate_probability](#calculate_probability), [domain_value_amount](#domain_value_amount)
# 
# * **training_data**: list[list[str]]: This is the data pulled that will build the model.
# * **smoothing**: bool: this value determines if smoothing will be used or not.
# 
# **returns**: list[tuple]: This returns a naive bayes classifier that will be used as a trained model to classify observations

# In[16]:


def train(training_data: list[list[str]], smoothing: bool=True):
    domains, nbc, label_totals_all = create_domains(training_data), [], {}
    class_labels = unique_attributes(domains[0])
    for label in class_labels:
        label_totals = domain_value_amount(training_data, 0, label)
        label_totals_all[label] = float(label_totals / len(training_data))
    nbc.append(label_totals_all)
    for i, domain in enumerate(domains[1:]):
        probabilities = {}
        for value in unique_attributes(domain):
            probabilities[value] = calculate_probability(training_data, (i+1), value, class_labels, smoothing, len(unique_attributes(domain)))
        nbc.append(probabilities)
    return nbc


# In[17]:


smoothed_unit_nbc = train(self_check)
smoothed_unit_nbc


# In[18]:


unit_nbc = train(self_check, False)
unit_nbc


# In[19]:


nbc = train(training_data)
nbc


# In[20]:


unsmoothed_nbc = train(training_data, False)
unsmoothed_nbc


# <a id="normalize"></a>
# ## normalize
# 
# This function is run after a probability is determined in order to make sure that the whole dictionary does in fact add up to 1 to see how likely a label is to be assigned **Used by**: [classify_instance](#classify_instance)
# 
# * **results**: dict:  The probability that is achieved from the nbc.
# * **class_labels**: list: the potential labels that can be assigned to the data
# 
# **returns**: dict: this returns a normalized dict that values add up to 1.

# In[21]:


def normalize(results: dict, class_labels: list):
    total, normalized_results = 0, {}
    for label in class_labels:
        total = total + results[label]
    for label in class_labels:    
        normalized_results[label] = results[label]/total
    return normalized_results


# In[22]:


unit_test = normalize({'y' : .1, 'n' : .04}, ['y', 'n'])
assert unit_test == {'y': 0.7142857142857143, 'n': 0.2857142857142857}
unit_test = normalize({'y' : .1, 'n' : .24}, ['y', 'n'])
assert unit_test == {'y': 0.2941176470588236, 'n': 0.7058823529411765}
unit_test = normalize({'y' : .01, 'n' : .00024}, ['y', 'n'])
assert unit_test == {'y': 0.9765624999999999, 'n': 0.0234375}


# <a id="find_best"></a>
# ## find_best
# 
# The purpose here is to find the best data after it has been calculated and normalized in order to select how it will be classified, as this is finding the best possible classifications **Used by**: [classify_instance](#classify_instance)
# 
# * **results**: dict:  The probability that is achieved from the nbc.
# * **class_labels**: list: the potential labels that can be assigned to the data
# 
# **returns**: dict: this returns the results that has been sorted to where the best option is the first option as well as the rest of the data to see much of a majority it was 

# In[23]:


def find_best(results: dict, class_labels: list):
    normalized = []
    for label in class_labels:
        normalized = [(results[label], label) for label in class_labels]
    return sorted(normalized, reverse=True)[0][1]


# In[24]:


unit_test = find_best({'y' : .6, 'n' : .4}, ['y', 'n'])
assert unit_test == 'y'
unit_test = find_best({'y' : .4444444, 'n' : .666666}, ['y', 'n'])
assert unit_test == 'n'
unit_test = find_best({'y' : 1, 'n' : 0}, ['y', 'n'])
assert unit_test == 'y'


# <a id="probability_of"></a>
# ## probability_of
# 
# This function is used to determine the probabilities of a given set of values for the features, as well as how likely each class label is given that set of data  **Used by**: [classify_instance](#classify_instance)
# 
# 
# * **nbc**: dict: this is the model that was built and being used to determine the class label.
# * **observation**: list: an instance that is being classified.
# * **label**: str: potential class labels.
# * **labeled**: bool: if the data is actually labeled or not, or if we need to determine the label, in this functions case it is for count in the array.
# 
# **returns**: float: this returns the specific probability of a label given a specific instance.

# In[25]:


def probability_of(nbc: list[dict], observation: list[str], label: str, labeled: bool):
    probability = nbc[0][label]
    for i in list(range(1, len(observation))):
        if labeled:
            probability = probability * nbc[i][observation[i]][label]
        else:
            probability = probability * nbc[i][observation[i-1]][label]
    return probability


# In[26]:


unit_test = probability_of(unit_nbc, ['s', 'l', 'r'], 'y', False)
assert unit_test == 0.14285714285714285
unit_test = probability_of(unit_nbc, ['s', 'l', 'b'], 'n', False)
assert unit_test == 0.08333333333333333
unit_test = probability_of(unit_nbc, ['r', 's', 'b'], 'n', False)
assert unit_test == 0.15000000000000002


# <a id="classify_instance"></a>
# ## classify_instance
# 
# The pseudocode given to this assignment actually resides here, where it takes in a specific instance of attributes to determine a class label given the model that has been built, using nbc, previously. It is designed to be called by classify so that it will be passed one instance at a time **Uses**: [probability_of](#probability_of), [find_best](#find_best), [normalize](#normalize) **Used by**: [classify](#classify)
# 
# * **nbc**: dict: this is the model that was built and being used to determine the class label.
# * **observation**: list: an instance that is being classified.
# * **class_labels**: list: potential class labels.
# * **labeled**: bool: if the data is actually labeled or not, or if we need to determine the label, in this functions case it is for count in the array.
# 
# 
# **returns**: tuple: returns the best value for the instance, as well as the results to see by what probability it was the best selection

# In[27]:


def classify_instance(nbc: list[dict], observation: list[str], class_labels: list[str], labeled: bool):
    results = {}
    for label in class_labels:
        results[label] = probability_of(nbc, observation, label, labeled)
    results = normalize(results, class_labels)
    best = find_best(results, class_labels)
    return (best, results)


# In[28]:


unit_test = classify_instance(unit_nbc, ['s', 'l', 'r'], ['y', 'n'], False)
assert unit_test[0][0] == 'y'
unit_test = classify_instance(unit_nbc, ['r', 's', 'g'], ['y', 'n'], False)
assert unit_test[0][0] == 'n'
unit_test = classify_instance(unit_nbc, ['r', 's', 'b'], ['y', 'n'], False)
assert unit_test[0][0] == 'n'


# <a id="classify"></a>
# ## classify
# 
# This function is essentially used to use classify to compare a list of observations that need to be labeled using the nbc model. **Uses**: [unique_attributes](#unique_attributes), [classify_instance](#classify_instance) **Used by**: [evaluate_model](#evaluate_model)
# 
# * **nbc**: dict: this is the model that was built and being used to determine the class label.
# * **observations**: list[list]: list of instances that is being classified.
# * **labeled**: bool: if the data is actually labeled or not, or if we need to determine the label, in this functions case it is for countint the array.
# * 
# **returns**: returns a list of the classifications of the instances for what class label is assigned to each instance

# In[29]:


def classify(nbc: dict, observations: list[list], labeled: bool=True):
    classifications = []
    if labeled:
        all_class_labels = [row[0] for row in observations] 
        class_labels = unique_attributes(all_class_labels)
    else:
        class_labels = ['e', 'p']
    for row in observations:
        classifications.append(classify_instance(nbc, row, class_labels, labeled))
    return classifications       


# In[30]:


unit_test = classify(unit_nbc, self_check)
assert unit_test[1][0] == 'y'
assert unit_test[2][0] == 'n'
assert unit_test[-1][0] == 'y'


# <a id="error_rate"></a>
# ## error_rate
# 
# This function takes the amount of errors, divided by the total, to determine the percent failed of the model
# 
# * **y**: list: list of correct answer
# * **y_hat**: list: list of guesses made by the model
# * **total**: int: the total amount in the set
# 
# **returns**: float: returns the percentage, in decimals, of errors found

# In[31]:


def error_rate(y: list, y_hat: list, total: int):
    error_rates = 0
    for i, correct in enumerate(y):
        if y_hat[i] != correct:
            error_rates += 1
    metric = error_rates/total
    return metric


# In[32]:


unit_test = error_rate([1, 2, 3, 1], [1, 1, 1, 1], 4)
assert unit_test == .5
unit_test = error_rate([1, 2, 3, 1], [1, 1, 3, 1], 4)
assert unit_test == .25
unit_test = error_rate([2, 2, 1, 4], [1, 1, 3, 1], 4)
assert unit_test == 1


# <a id="evaluate"></a>
# ## evaluate
# 
# This is the function that determines what the actual class labels are, as well as determines what the predicted class labels, and compares them to determine the accuracy of the model **Uses**: [classify](#classify) **Used by**: [cross_validation](#cross_validation)
# 
# * **data**: list[list]: This is the data that is being tested against, used for testing a model and training it.
# * **model**: list: This is the model that is being tested.
# * **evaluation_fn**: function: this function is used to actually compare the answer vs what the model got as a class label
# * **classify_fn**: function: this is the function that is used to classify the data given the model and the data
# 
# **returns**: float: this function returns the error rate that was achieved by comparing the model to the actuals

# In[33]:


def evaluate(data: list[list], model: list, evaluation_fn, classify_fn):
    y_hats, ys = [], []
    for row in data:
        ys.append(row[0])
    y_hats_results = classify_fn(model, data)
    for row in y_hats_results:
        y_hats.append(row[0])
    metric = evaluation_fn(ys, y_hats, len(y_hats))
    return metric


# In[34]:


unit_test = evaluate(self_check, unit_nbc, error_rate, classify)
assert unit_test == 0.26666666666666666
unit_test = evaluate(self_check[2:], unit_nbc, error_rate, classify)
assert unit_test == 0.3076923076923077
unit_test = evaluate(self_check[4:], unit_nbc, error_rate, classify)
assert unit_test == 0.2727272727272727


# <a id="default_poison"></a>
# ## default_poison
# 
# Only use this function has is to just make a list of desired length all to be assigned to the default chosen for testing a baseline
# 
# * **length**: int: the length of the string desired to make
# * **default**: str:  the value to set to the list
# 
# **returns**: list: returns a list of the default str 

# In[35]:


def default_poison(length: int, default: str):
    baseline = []
    for i in list(range(length)):
        baseline.append(default)
    return baseline


# In[36]:


unit_test = default_poison(5, 'a')
assert unit_test == ['a', 'a', 'a', 'a', 'a']
unit_test = default_poison(6, 'h')
assert unit_test == ['h', 'h', 'h', 'h', 'h', 'h']
unit_test = default_poison(6, 'p')
assert unit_test == ['p', 'p', 'p', 'p', 'p', 'p']


# <a id="baseline_cross_validation"></a>
# ## baseline_cross_validation
# 
# This function purely sets up a baseline comparison for the 10 fold cross validation that uses the defaul p for every single guess to establish a baseline to test against **Uses**: [error_rate](#error_rate), [evaluate_model](#evaluate_model) 
# 
# * **folds: List[List[List[float]]]**: here is the set of data that will be tested against, broken into 10 folds
# 
#   
# **returns**: float: this function returns the error calculation of the test set and model, the ys and y_hats

# In[37]:


def baseline_cross_validation(folds: list[list[list]]):
    train_metrics, test_metrics, n = [], [], len(folds)
    for i in range(0, n):
        train, test = create_train_test(folds, i)
        baseline_train_array, baseline_test_array = default_poison(len(train), 'p'), default_poison(len(test), 'p')
        train_ys, test_ys = [], []
        for row in train:
            train_ys.append(row[0])
        for row in test:
            test_ys.append(row[0])
        metric = error_rate(train_ys, baseline_train_array, len(train))
        train_metrics.append(metric)
        metric = error_rate(test_ys, baseline_test_array, len(test))
        test_metrics.append(metric)
    return train_metrics, test_metrics


# In[38]:


unit_fold = create_folds(self_check, 3)
unit_test = baseline_cross_validation(unit_fold)
assert unit_test == ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
unit_fold = create_folds(self_check, 5)
unit_test = baseline_cross_validation(unit_fold)
assert unit_test == ([1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0])
unit_fold = create_folds(self_check, 4)
unit_test = baseline_cross_validation(unit_fold)
assert unit_test == ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])


# <a id="cross_validation"></a>
# ## cross_validation
# 
# The purpose of this function is to use 10 crossfold validation by taking in the data as a whole, and also the folds created by create_folds, to have a test set created out of one of the 10 folds that is used to test the other 9, which then cycles down until each of the folds is used to validate the other data **Uses**: [mean_squared_error](#mean_squared), [evaluate_model](#evaluate_model) 
# 
# * **folds: List[List[List[float]]]**: here is the set of data that will be tested against, broken into 10 folds
# * **model_fn: function: float** This is the baseline model that will be tested against to determine error.
# * **evaluation_fn: function: float**: this is the function that is used to test how the model is doing vs the answers, for here it is mean squared error.
# * **classify_fn**: function: this is the function that is used to classify the data given the model and the data
# 
#   
# **returns**: float: this function returns the error calculation of the test set and model, the ys and y_hats

# In[39]:


def cross_validation(folds, model_fn, evaluation_fn, classify_fn): 
    train_metrics, test_metrics, n = [], [], len(folds)
    for i in range(0, n):
        train, test = create_train_test(folds, i)
        model = model_fn(train)
        metric = evaluate(train, model, evaluation_fn, classify_fn)
        train_metrics.append(metric)
        metric = evaluate(test, model, evaluation_fn, classify_fn)
        test_metrics.append(metric)
    return train_metrics, test_metrics


# In[40]:


unit_train, unit_test = cross_validation(unit_folds, lambda unit_nbc: train(self_check), error_rate, classify)
assert unit_test == [0.0,
 0.3333333333333333,
 0.3333333333333333,
 0.3333333333333333,
 0.3333333333333333]
assert unit_train == [0.3333333333333333, 0.25, 0.25, 0.25, 0.25]
unit_test, unit_train = cross_validation(unit_folds, lambda unit_nbc: train(self_check), error_rate, classify)
assert unit_train == [0.0,
 0.3333333333333333,
 0.3333333333333333,
 0.3333333333333333,
 0.3333333333333333]
assert unit_test == [0.3333333333333333, 0.25, 0.25, 0.25, 0.25]


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

# In[41]:


def analyzation(train_metrics, test_metrics):
    train_avg, train_std, test_avg, test_std = np.mean(train_metrics), np.std(train_metrics), np.mean(test_metrics), np.std(test_metrics)
    test_avg = np.mean(test_metrics)
    print(f"Fold\tTrain\tTest")
    for i in range(len(train_metrics)):
        train_metrics[i] = train_metrics[i] * 100
        test_metrics[i] = test_metrics[i] * 100
        print(f"{i+1}: \t{train_metrics[i]:.2f}%\t{test_metrics[i]:.2f}%")
    print("\n\n")
    train_avg = train_avg * 100
    test_avg = test_avg * 100
    print(f"avg:\t{train_avg:.2f}%\t{test_avg:.2f}%")
    print(f"std:\t{train_std:.2f}%\t{test_std:.2f}%")


# In[42]:


baseline_train, baseline_test = baseline_cross_validation(folds)
nbc_train, nbc_test = cross_validation(folds, lambda nbc: train(training_data), error_rate, classify)
unsmoothed_nbc_train, unsmoothed_nbc_test = cross_validation(folds, lambda nbc: train(training_data, False), error_rate, classify)
print('Baseline')
analyzation(baseline_train, baseline_test)
print('Smoothed')
analyzation(nbc_train, nbc_test)
print('Unsmoothed')
analyzation(unsmoothed_nbc_train, unsmoothed_nbc_test)


# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# Here is my analysis of the model and the data. I think I was surprised to see how accurate this model was, and it was consistently fairly accurate as every time I reran it, the results in the cross validation were incredibly similar. I almost feel like I did something wrong, or right I guess? For a baseline algorithm that is typically seen as one of the easier ones, that seems like a tough baseline to beat. 
# 
# The other interesting observation I have is that by looking at the data for a smoothed comparison and an unsmoothed run of the nbc, I was surprised to see how incredibly acccurate the unsmoothed variation was. I wonder if my smoothing function added to much data when I went through using that, or if that is suppose to happen, but it consistently happened to where the unsmoothed was much, much more accurate, but both were still very accurate.
# 
# Another note, is that I saw your note on ignoreing the '?'s but I did not get to that in time, and I apologize for that.
# 
# Thanks again and let me know anything that I did wrong or need to do again here.

# In[ ]:




