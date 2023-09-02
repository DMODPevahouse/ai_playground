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
