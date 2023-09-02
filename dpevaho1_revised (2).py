#!/usr/bin/env python
# coding: utf-8

# # Module 8 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# In[1]:


from copy import deepcopy
import random
from typing import List, Dict, Tuple, Callable
import math
from operator import itemgetter
import pprint
import numpy as np
import matplotlib.pyplot as plt


# ## Decision Trees
# 
# For this assignment you will be implementing and evaluating a Decision Tree using the ID3 Algorithm (**no** pruning or normalized information gain). Use the provided pseudocode. The data is located at (copy link):
# 
# http://archive.ics.uci.edu/ml/datasets/Mushroom
# 
# **Just in case** the UCI repository is down, which happens from time to time, I have included the data and name files on Blackboard.
# 
# <div style="background: lemonchiffon; margin:20px; padding: 20px;">
#     <strong>Important</strong>
#     <p>
#         No Pandas. The only acceptable libraries in this class are those contained in the `environment.yml`. No OOP, either. You can used Dicts, NamedTuples, etc. as your abstract data type (ADT) for the the tree and nodes.
#     </p>
# </div>
# 
# One of the things we did not talk about in the lectures was how to deal with missing values. There are two aspects of the problem here. What do we do with missing values in the training data? What do we do with missing values when doing classifcation?
# 
# For the first problem, C4.5 handled missing values in an interesting way. Suppose we have identifed some attribute *B* with values {b1, b2, b3} as the best current attribute. Furthermore, assume there are 5 observations with B=?, that is, we don't know the attribute value. In C4.5, those 5 observations would be added to *all* of the subsets created by B=b1, B=b2, B=b3 with decreased weights. Note that the observations with missing values are not part of the information gain calculation.
# 
# This doesn't quite help us if we have missing values when we use the model. What happens if we have missing values during classification? One approach is to prepare for this advance. When you train the tree, you need to add an implicit attribute value "?" at every split. For example, if the attribute was "size" then the domain would be ["small", "medium", "large", "?"]. The "?" value gets all the data (because ? is now a wildcard). However, there is an issue with this approach. "?" becomes the worst possible attribut value because it has no classification value. What to do? There are several options:
# 
# 1. Never recurse on "?" if you do not also recurse on at least one *real* attribute value.
# 2. Limit the depth of the tree.
# 
# There are good reasons, in general, to limit the depth of a decision tree because they tend to overfit.
# Otherwise, the algorithm *will* exhaust all the attributes trying to fulfill one of the base cases.
# 
# You must implement the following functions:
# 
# `train` takes training_data and returns the Decision Tree as a data structure. There are many options including namedtuples and just plain old nested dictionaries. **No OOP**.
# 
# ```
# def train(training_data, depth_limit=None):
#    # returns the Decision Tree.
# ```
# 
# The `depth_limit` value defaults to None. (What technique would we use to determine the best parameter value for `depth_limit` hint: Module 3!)
# 
# `classify` takes a tree produced from the function above and applies it to labeled data (like the test set) or unlabeled data (like some new data).
# 
# ```
# def classify(tree, observations, labeled=True):
#     # returns a list of classifications
# ```
# 
# `evaluate` takes a data set with labels (like the training set or test set) and the classification result and calculates the classification error rate:
# 
# $$error\_rate=\frac{errors}{n}$$
# 
# Do not use anything else as evaluation metric or the submission will be deemed incomplete, ie, an "F". (Hint: accuracy rate is not the error rate!).
# 
# `cross_validate` takes the data and uses 10 fold cross validation (from Module 3!) to `train`, `classify`, and `evaluate`. **Remember to shuffle your data before you create your folds**. I leave the exact signature of `cross_validate` to you but you should write it so that you can use it with *any* `classify` function of the same form (using higher order functions and partial application).
# 
# Following Module 3's discussion, `cross_validate` should print out the fold number and the evaluation metric (error rate) for each fold and then the average value (and the variance). What you are looking for here is a consistent evaluation metric cross the folds. You should print the error rates in terms of percents (ie, multiply the error rate by 100 and add "%" to the end).
# 
# ```
# def pretty_print_tree(tree):
#     # pretty prints the tree
# ```
# 
# This should be a text representation of a decision tree trained on the entire data set (no train/test).
# 
# To summarize...
# 
# Apply the Decision Tree algorithm to the Mushroom data set using 10 fold cross validation and the error rate as the evaluation metric. When you are done, apply the Decision Tree algorithm to the entire data set and print out the resulting tree.
# 
# **Note** Because this assignment has a natural recursive implementation, you should consider using `deepcopy` at the appropriate places.
# 
# -----

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
    return data


# In[3]:


data = parse_data("agaricus-lepiota.data")
assert data[0] == ['p', 'x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u']
assert len(data) == 8124
assert len(data[0]) == 23


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


folds = create_folds(data, 10)

assert len(folds) == 10
assert len(folds[0]) == 813
assert len(folds[9]) == 812


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


train, test = create_train_test(folds, 0)
assert len(train) == 7311
assert len(test) == 813
assert len(train[0]) == 23


# <a id="get_count"></a>
# ## get_count
# 
# This function takes in the attribute in a column as well as the domain of that column and determines how many of each values of the domain reside in the column. In this case, only one value from the domain will be tested when it is called  **Used by**: [homogeneous](#homogeneous), [get_majority_label](#get_majority_label), [pick_best_attribute](#pick_best_attribute)
# 
# * **attributes**: list[str]: a list of values that are within the domain.
# * **domain**: str: A specific value in the domain to determine how many are in the count.
# 
# 
# **returns**: int, returns a number of how many 

# In[8]:


def get_counts(attributes: list[str], domain: str):
    count = 0
    for attribute in attributes:
        if attribute == domain:
            count += 1
    return count
            


# In[9]:


unit_count = get_counts([1, 1, 2, 3, 1], 1)
assert unit_count == 3
unit_count = get_counts(['a', 'a', 'z', 'd', 'a', 'a', 'z'], 'z')
assert unit_count == 2
unit_count = get_counts(['a', 'a', 'z', 'd', 'a', 'a', 'z'], 'd')
assert unit_count == 1


# <a id="unique_attributes"></a>
# ## unique_attributes
# 
# This function looks at a list of data to determine how many unique values are in there to be tested and create a domain  **Used by**: [id3](#id3), [homogeneous](#homogeneous), [get_majority_label](#get_majority_label), [pick_best_attribute](#pick_best_attribute)
# 
# * **attributes**: list:  A list attributes to be used to determine how many unique values there are .
# 
# 
# **returns**: list: returns a list that are unique from the attributes provided

# In[10]:


def unique_attributes(attributes: list[str]):
    unique = []
    for a in attributes:
        if a not in unique:
            unique.append(a)
    return unique


# In[11]:


unit_test_list = [1, 1, 1, 2, 2, 2, 3, 3 ,3]
unit_test = unique_attributes(unit_test_list)
assert unit_test == [1, 2, 3]
unit_test_list = ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'z' , 'z']
unit_test = unique_attributes(unit_test_list)
assert unit_test == ['a', 'b', 'c', 'z']
unit_test_list = ['a', 'a', 1, 1, 2, 2, 'z', 'z' , 'z']
unit_test = unique_attributes(unit_test_list)
assert unit_test == ['a', 1, 2, 'z']


# <a id="available_domains"></a>
# ## available_domains
# 
# This list is used to create a list of objects that are in the column being tested from the available attributes list. So it takes a column, the data, and creates the data from that information **Used by**: [id3](#id3), [homogeneous](#homogeneous), [get_majority_label](#get_majority_label), [pick_best_attribute](#pick_best_attribute)
# 
# * **data**: list[list]: the data to grab from the column.
# * **col**: int: the column that the data is needed from.
# * (add more as necessary)
# 
# **returns**: list: returns the column data using the original data and the column

# In[12]:


def available_domains(data: list[list], col: int):
    best_attribute_data = []
    for row in data:
        if len(row) > col:
            best_attribute_data.append(row[col])
    return best_attribute_data
    


# In[13]:


unit_test = available_domains(test, 1)
assert len(unit_test) == 813
unit_test = available_domains(train, 1)
assert len(unit_test) == 7311
unit_test = available_domains(data, 20)
assert len(unit_test) == 8124


# <a id="calculate_entropy"></a>
# ## calculate_entropy
# 
# This function uses the entropy calculation to dermine the specific entropy from the attribute data to determine if it will be helpful in creating a better tree or not **Used by**:  [homeogeneous](#homeogeneous), [pick_best_attribute](#pick_best_attribute)
# 
# * **total**: int: the total in the data being calculated.
# * **counts**: list: list of how often domains appear in the data.
# 
# 
# **returns**: float: returns the calculated entropy.

# In[14]:


def calculate_entropy(total: int, counts: list):
    sum = 0
    for x in counts:
        if x != 0:
            calculation = -((float(x)/total) * math.log2(float(x)/total))
            sum = sum + calculation
    return sum


# In[15]:


unit_test = calculate_entropy(100, [16, 70, 3, 4])
assert unit_test == 1.1207392696073828
unit_test = calculate_entropy(100, [1, 7, 30, 41])
assert unit_test == 1.3834680448020302
unit_test = calculate_entropy(100, [1, 7, 3, 4, 1, 10, 8, 9])
assert unit_test == 1.6753083824189203


# <a id="homogeneous"></a>
# ## homogeneous
# 
# This function uses calculate_entropy to determine if the entropy is perfectly 0, so homogenous, and if it is not, nothing will be returned **Uses**: [calculate_entropy](#calculate_entropy), [available_domain](#available_domain), [unique_attributes](#unique_attribute), [get_counts](#get_counts). **Used by**: [id3](#id3)
# 
# * **data**: list[list]:  A list of lists that will be tested to see if the values are homogenous or not.
# 
# 
# **returns**: returns None if the data is not homogenous, returns 0.0 otherwise

# In[16]:


def homogeneous(data: list[list]):
    domain_possible = available_domains(data, 0)
    domain = unique_attributes(domain_possible)
    total = len(data)
    counts = []
    for value in domain:
        counts.append(get_counts(domain_possible, value))
    e = calculate_entropy(total, counts)
    if e == 0.0:
        return e


# In[17]:


unit_test = homogeneous(data)
assert unit_test == None
unit_test = homogeneous([[1, 0], [1, 0]])
assert unit_test == 0.0
unit_test = homogeneous(test)
assert unit_test == None


# <a id="reduce_data"></a>
# ## reduce_data
# 
# Description of what the function does (in prose) and the significance of the function if it is part of the actual algorithm (and not just a helper function). **Used by**: [id3](#id3)
# 
# * **data**: list[list]: data that will be reduced
# * **best_attribute**: str, the best attribute that will be compared with to determine how the data will be reduced
# * **value**: int: the column that will be compared to determine if the data is kept or not 
# 
# **returns**: list[list]: returns the reduced data

# In[18]:


def reduce_data(data: list[list], best_attribute: str, value: int):
    edited_data = []
    for row in data:
        if row[best_attribute] == value:
            edited_data.append(row)
    return edited_data


# In[19]:


unit_test = reduce_data(deepcopy(data), 0, 'p')
assert unit_test[0] == ['p', 'x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u']
unit_test = reduce_data(deepcopy(data), -1, 'u')
assert unit_test[10] == ['e', 'x', 'f', 'n', 'f', 'n', 'f', 'c', 'n', 'g', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 'y', 'u']
unit_test = reduce_data(deepcopy(data), 8, 'b')
assert unit_test[11] == ['e', 'f', 'f', 'w', 'f', 'n', 'f', 'w', 'b', 'k', 't', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'e', 'n', 'a', 'g']


# <a id="shrink_attributes"></a>
# ## shrink_attributes
# 
# A simple purpose, this function just removes an elemnt from the list of attributes so that it is not checked again in a tree **Used by**: [id3](#id3).
# 
# * **attributes**: list: list of attributes that will have a value removed.
# * **col**: int: the location to be removed.
# 
# **returns**: list: returns the attribute list without the specified location

# In[20]:


def shrink_attributes(attributes: list, col: int):
    attributes.pop(col)
    return attributes


# In[21]:


unit_test = shrink_attributes(list(range(0, 23)), 9)
assert unit_test == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
unit_test = shrink_attributes(unit_test, 5)
assert unit_test == [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
unit_test = shrink_attributes(unit_test, 0)
assert unit_test == [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


# <a id="majority_label"></a>
# ## majority_label
# 
# This function takes a look at the specified column of data to determine which attribute used is the most common, while at the same time looks at what the appropriate class lable for that data would be **Uses**: [available_domain](#available_domain), [unique_attributes](#unique_attribute), [get_counts](#get_counts). **Used by**: [id3](#id3)
# 
# * **data**: list[list]: this is the data that will be used to determine what the class label of a specific attribute is
# * **col**: int: this is the location of the attribute that we are finding a class label and the majority attribute
# 
# **returns**: str, str: returns the majority label of the attribute as well as what the most common class label is of that majority

# In[22]:


def majority_label(data: list, col: int):
    domain_possible, class_domain_possible = available_domains(data, col), available_domains(data, 0)
    domain, class_domain = unique_attributes(domain_possible), unique_attributes(class_domain_possible)
    domain_frequencies, class_frequencies = [], []
    for value in domain:
        count = get_counts(domain_possible, value)
        domain_frequencies.append((value, count))
    majority_attribute = sorted(domain_frequencies, key=itemgetter(1), reverse=True)
    domain = unique_attributes(domain_possible) 
    for value in class_domain:
        count = get_counts(class_domain_possible, value)
        class_frequencies.append((value, count))
    class_labels = sorted(class_frequencies, key=itemgetter(1), reverse=True)
    return majority_attribute[0][0], class_labels[0][0]


# In[23]:


unit_test = majority_label(deepcopy(test), 1)
assert unit_test == ('x', 'e')
unit_test = majority_label(deepcopy(test), 20)
assert unit_test == ('n', 'e')
unit_test = majority_label(deepcopy(train), 9)
assert unit_test == ('b', 'p')


# <a id="pick_best_attribute"></a>
# ## pick_best_attribute
# 
# This function is used to go through all of the available attribute and the data it is used in to determine which attribute should be the next that adds the most useful information. It uses both entropy and information gain to make that determination **Uses**: [available_domain](#available_domain), [unique_attributes](#unique_attribute), [get_counts](#get_counts), [calculate_entropy](#calculate_entropy)  **Used by**: [id3](#id3)
# 
# * **data**: list[list]: this is the data that will be used to determine what the class label of a specific attribute is
# * **attributes**: list: remaining attributes to be evalued .
# * **col**: int: this is the of the attributes that is being evaluated to determine if it is the best pick
# 
# **returns**: tuple: this tuple returns the value of what the attribute is as well as the column that the attribute resides in

# In[24]:


def pick_best_attribute(data: list[list], attributes: list, col: int):
    potential_best, counts = [], []
    class_domain_possible = available_domains(data, 0)
    class_domain = unique_attributes(class_domain_possible)
    for value in class_domain:
        counts.append(get_counts(class_domain_possible, value))
    target_entropy = calculate_entropy(len(data), counts)
    potential_best.append([attributes[0], 0, target_entropy])
    for i, attribute in enumerate(attributes):
        if attribute != col:
            domain_possible, counts = available_domains(data, attribute), []
            domain = unique_attributes(domain_possible)
            for value in domain:
                counts.append(get_counts(domain_possible, value))
            entropy = calculate_entropy(len(data), counts)
            information_gain = target_entropy - entropy
            potential_best.append([attribute, i, information_gain])
    actual_best = sorted(potential_best, key=itemgetter(2), reverse=True)
    return actual_best[0][0]


# In[25]:


unit_test = pick_best_attribute(test, list(range(1,10)), 0)
assert unit_test == 1
unit_test = pick_best_attribute(data, list(range(8,23)), 7)
assert unit_test == 8
unit_test = pick_best_attribute(data, list(range(15,23)), 7)
assert unit_test == 15


# <a id="train"></a>
# ## train
# 
# This is the main algorithm that is being used. It uses the id3 algorithm to train the data. The goal here is to pick the best attributes to build a model that will be used to determine a classification based on the testing and training data to be used. Other functions are used to determine which attribute will be selected next to be used, while shrinking the data and attributes so that the algorithm can recursivily call itself in order to build a tree so that based on data that is being tested can find correct classifications **Uses**:[available_domain](#available_domain), [unique_attributes](#unique_attribute), [get_counts](#get_counts), [shrink_attributes](#shrink_attributes), [majority_label](#majority_label), [pick_best_attributes](#pick_best_attributes)
# 
# * **data**: list[list]: the data that will be evaluated
# * **attributes**: list: the columns of data that are available to be picked in order to continue
# * **col**: int: a starting default column to for the algorithm to have a starting place
# * **default**: str: this is in case the algorithm cannot find the answer, so instead of having a random suggestion, a safer option would be selected
# * **tree**: dict: the tree that is being built as the model
# * **depth_limit**: int: how far down the tree will go when being built, this is set to avoid overfitting so that the tree does not be built past a certain depth
# 
# **returns**: dict: this returns a dictionary tree that will be used as a model to determine classifications

# In[26]:


def id3(data: list[list], attributes: list[int], col: int, default: str, tree: dict, depth_limit: int=None):
    if len(data) == 0:  return default
    attribute_value, classification = majority_label(deepcopy(data), col)
    if homogeneous(data) == 0.0: return {attribute_value: classification}
    if attributes is [] or depth_limit == 0: return {attribute_value: classification}
    best_attribute = pick_best_attribute(data, attributes, col)
    tree = {attribute_value: classification}
    for j, best in enumerate(attributes):
        if best == best_attribute:
            for i, value in enumerate(unique_attributes(available_domains(deepcopy(data), attributes[j]))):
                if value == '?' and i == 0: continue
                else:
                    subset = reduce_data(deepcopy(data), attributes[j], value)
                    if depth_limit is not None: child = id3(subset, shrink_attributes(deepcopy(attributes), j), i, value, tree, depth_limit-1)
                    else:  child = id3(subset, shrink_attributes(deepcopy(attributes), j), i, value, tree, depth_limit)
                    tree[value] = child
    return tree


# In[27]:


tree = id3(train, list(range(1,23)), 1, 'p', {})


# <a id="print_the_tree"></a>
# ## print_the_tree
# 
# This function simply takes in a dictionary and prints it in a matter that is easy to follow how it acts like a tree
# 
# * **tree**: dict: the model that was built to be printed
# 
# **returns**: None: just prints the tree

# In[28]:


def print_the_tree(tree):
    pp = pprint.PrettyPrinter()
    pp.pprint(tree)


# In[29]:


print_the_tree(tree)


# <a id="classify"></a>
# ## classify
# 
# This function is a recursive algorithm that is built to use the model. It takes the model that was built previously and goes through to determine the answer. Since the model was built as a tree in a dictionary, if the value brings up a dictionary, it is removed as well as recursively going through until a proper answer is found
# 
# * **model**: dict: the tree that was built to be used as a classification
# * **attributes_in_question**: list: the attributes given to determine what is being described
# * **default**: str: a default to fall back to if there is not a clear answer
# 
# **returns**: str: returns the classification found for the set in question

# In[30]:


def classify(model: dict, attributes_in_question: list, default: str):
    for i, attribute in enumerate(attributes_in_question):
        if attribute in model:
            if isinstance(model[attribute], dict):
                next_attributes = deepcopy(attributes_in_question)
                next_attributes.pop(i)
                return classify(model[attribute], next_attributes, default)
            elif isinstance(model[attribute], str):
                return model[attribute]
        else:
            continue
    return default


# In[31]:


unit_test = classify({'i': 'a', 's': 'a'}, ['a'], 'x')
assert unit_test == 'x'
unit_test = classify({'i': 'a', 's': 'a'}, ['i'], 'x')
assert unit_test == 'a'
unit_test = classify({'i': 'a', 's': 'b'}, ['s'], 'x')
assert unit_test == 'b'


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

# In[32]:


def error_rate(y: list, y_hat: list, total: int):
    error_rates = 0
    for i, correct in enumerate(y):
        if y_hat[i] != correct:
            error_rates += 1
    metric = error_rates/total
    return metric


# In[33]:


unit_test = error_rate([1, 2, 3, 1], [1, 1, 1, 1], 4)
assert unit_test == .5
unit_test = error_rate([1, 2, 3, 1], [1, 1, 3, 1], 4)
assert unit_test == .25
unit_test = error_rate([2, 2, 1, 4], [1, 1, 3, 1], 4)
assert unit_test == 1


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

# In[34]:


def evaluate_model(data, model, evaluation_fn):
    y_hats, ys = [], []
    for row in data:
        y_hats.append(classify(model, row[1:], 'p'))
        ys.append(row[0])
    metric = evaluation_fn(ys, y_hats, len(y_hats))
    return metric


# In[35]:


test1 = evaluate_model(test, tree, error_rate)
assert test1 == 0.21279212792127922
test1 = evaluate_model(train, tree, error_rate)
assert test1 == 0.18725208589796197
test1 = evaluate_model(data, tree, error_rate)
assert test1 == 0.18980797636632202


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

# In[36]:


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


# In[37]:


unit_attributes = list(range(1, 23))
unit_train1, unit_test1 = cross_validation(folds, lambda tree: id3(data, unit_attributes, 1, 'p', {}, depth_limit=None), error_rate)
assert unit_train1 == [0.1598960470523868, 0.16523047462727397, 0.14498700588154836, 0.1318561072356723, 0.15207877461706784, 0.1786105032822757, 0.17669584245076586, 0.1801148796498906, 0.1823030634573304, 0.1801148796498906]
assert unit_test1 == [0.21279212792127922, 0.16482164821648215, 0.34686346863468637, 0.46494464944649444, 0.2832512315270936, 0.04433497536945813, 0.06157635467980296, 0.03078817733990148, 0.011083743842364532, 0.03078817733990148]
unit_train2, unit_test2 = cross_validation(folds, lambda tree: id3(data, unit_attributes, 1, 'p', {}, depth_limit=3), error_rate)
assert unit_train2 == [0.32540008206811655, 0.3228012583777869, 0.32307481876624267, 0.3159622486663931, 0.3057986870897155, 0.33533916849015316, 0.324535010940919, 0.31797045951859954, 0.3298687089715536, 0.33533916849015316]
assert unit_test2 == [0.3075030750307503, 0.33087330873308735, 0.3284132841328413, 0.3923739237392374, 0.4839901477832512, 0.21798029556650247, 0.31527093596059114, 0.37438423645320196, 0.2672413793103448, 0.21798029556650247]
unit_train3, unit_test3 = cross_validation(folds, lambda tree: id3(data, unit_attributes, 1, 'p', {}, depth_limit=5), error_rate)
assert unit_train3 == [0.15866502530433593, 0.16358911229653947, 0.14211462180276296, 0.1282998221857475, 0.14852297592997812, 0.17464442013129103, 0.17218271334792123, 0.17546498905908095, 0.17902078774617067, 0.1773796498905908]
assert unit_test3 == [0.1918819188191882, 0.14760147601476015, 0.34071340713407133, 0.46494464944649444, 0.2832512315270936, 0.0480295566502463, 0.07019704433497537, 0.04064039408866995, 0.008620689655172414, 0.023399014778325122]


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

# In[38]:


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


# In[39]:


analyzation(unit_train1, unit_test1)
analyzation(unit_train2, unit_test2)
analyzation(unit_train3, unit_test3)


# <a id="default_poison"></a>
# ## default_poison
# 
# Only use this function has is to just make a list of desired length all to be assigned to the default chosen for testing a baseline
# 
# * **length**: int: the length of the string desired to make
# * **default**: str:  the value to set to the list
# 
# **returns**: list: returns a list of the default str 

# In[40]:


def default_poison(length: int, default: str):
    baseline = []
    for i in list(range(length)):
        baseline.append(default)
    return baseline


# In[41]:


unit_test = default_poison(5, 'a')
assert unit_test == ['a', 'a', 'a', 'a', 'a']
unit_test = default_poison(6, 'h')
assert unit_test == ['h', 'h', 'h', 'h', 'h', 'h']
unit_test = default_poison(6, 'p')
assert unit_test == ['p', 'p', 'p', 'p', 'p', 'p']


# id="baseline_cross_validation"></a>
# ## baseline_cross_validation
# 
# This function purely sets up a baseline comparison for the 10 fold cross validation that uses the defaul p for every single guess to establish a baseline to test against **Uses**: [error_rate](#error_rate), [evaluate_model](#evaluate_model) 
# 
# * **folds: List[List[List[float]]]**: here is the set of data that will be tested against, broken into 10 folds
# 
#   
# **returns**: float: this function returns the error calculation of the test set and model, the ys and y_hats

# In[42]:


def baseline_cross_validation(folds):
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


# In[43]:


attributes = list(range(1, 23))
baseline_train, baseline_test = baseline_cross_validation(folds)
id3_train, id3_test = cross_validation(folds, lambda tree: id3(data, attributes, 1, 'p', {}, depth_limit=None), error_rate)
analyzation(baseline_train, baseline_test)
analyzation(id3_train, id3_test)


# In[44]:


train, test = create_train_test(folds, 0)
train_depth = []
test_depth = []
for depth in range(1, 40):
    _id3 =  id3(train, list(range(1,23)), 1, 'p', {}, depth)
    metric = evaluate_model(train, _id3, error_rate)
    train_depth.append(metric*100)
    metric = evaluate_model(test, _id3, error_rate)
    test_depth.append(metric*100)
depths = list(range(1, 40))
plt.plot(depths, train_depth, 'b', depths, test_depth, 'r')
plt.xlabel('depth')
plt.ylabel('Error')
plt.gca().invert_xaxis()
plt.show();


# In[45]:


for depth in list(range(1,13)):
    baseline_train, baseline_test = baseline_cross_validation(folds)
    id3_train, id3_test = cross_validation(folds, lambda tree: id3(data, attributes, 1, 'p', {}, depth_limit=depth), error_rate)
    analyzation(baseline_train, baseline_test)
    analyzation(id3_train, id3_test)


# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# 

# I finished this assignment but I have a few notes of what I think I did wrong. I believe, as long as I was analyzing my model correctly, that my error rate stuck towards 15%. I remember during office hours you mentioned that it should be lower then 10% but I never obtained that. My guess is, based off some of my unit tests and trouble shooting that the issue was centered around my pick_best_attribute function, as it seemed to have problems picking the column, though by the time I realized that I was significantly nearly out of time unfortunately. That is however where I think my issues were with not reaching 10%
# 
# Another oddity was that the depth did not seem to change how my model worked at all. Based off of the evaluation of it in the graph, having extremely small depth limits of course effected it, but after that there was not much more to be effected. That may have had to do with my classify? I am not sure entirely. 
# 
# One last note, was that the directions said to keep the name of the function to be train, but I left it as id3 for a couple reasons. One, which is on me, I did not pay attention to that detail until well into the assignment, so I would have a lot of rework to do to go and find it. And two, I wanted to keep the consistency for the training set to be called train, like in module 3. 

# TLDR: Problem functions were probably pick_best_attribute and classify
# 
# Depth did not seem to affect anything, not sure why, possibly classify
# 
# Train is not train but id3

# FYI: I restarted the kernel to run everything again, and the model accuracy dropped from 15% to 30%. I still expect the same culprits but not sure what I had floating around in the kernel that caused that. 
# 
# I apologize.

# NEW
# 
# 
# 
# Through some revisions I was able to bring the error rate from 35% to 15%, which I know is still not perfect but it was a lot closer. I did this by reworking by pick_best_attribute and using some variables correctly in the id3 model that was not making a ton of sense when I went back and looked at it so that it was reducing and using the right column and values

# In[ ]:




