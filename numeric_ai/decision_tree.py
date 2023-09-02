from copy import deepcopy
import random
from typing import List, Dict, Tuple, Callable
import math
from operator import itemgetter
import pprint
import numpy as np
import matplotlib.pyplot as plt


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


unit_test_list = [1, 1, 1, 2, 2, 2, 3, 3, 3]
unit_test = unique_attributes(unit_test_list)
assert unit_test == [1, 2, 3]
unit_test_list = ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'z', 'z']
unit_test = unique_attributes(unit_test_list)
assert unit_test == ['a', 'b', 'c', 'z']
unit_test_list = ['a', 'a', 1, 1, 2, 2, 'z', 'z', 'z']
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
            calculation = -((float(x) / total) * math.log2(float(x) / total))
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
assert unit_test[0] == ['p', 'x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o',
                        'p', 'k', 's', 'u']
unit_test = reduce_data(deepcopy(data), -1, 'u')
assert unit_test[10] == ['e', 'x', 'f', 'n', 'f', 'n', 'f', 'c', 'n', 'g', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o',
                         'p', 'k', 'y', 'u']
unit_test = reduce_data(deepcopy(data), 8, 'b')
assert unit_test[11] == ['e', 'f', 'f', 'w', 'f', 'n', 'f', 'w', 'b', 'k', 't', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o',
                         'e', 'n', 'a', 'g']


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


unit_test = pick_best_attribute(test, list(range(1, 10)), 0)
assert unit_test == 1
unit_test = pick_best_attribute(data, list(range(8, 23)), 7)
assert unit_test == 8
unit_test = pick_best_attribute(data, list(range(15, 23)), 7)
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


def id3(data: list[list], attributes: list[int], col: int, default: str, tree: dict, depth_limit: int = None):
    if len(data) == 0:  return default
    attribute_value, classification = majority_label(deepcopy(data), col)
    if homogeneous(data) == 0.0: return {attribute_value: classification}
    if attributes is [] or depth_limit == 0: return {attribute_value: classification}
    best_attribute = pick_best_attribute(data, attributes, col)
    tree = {attribute_value: classification}
    for j, best in enumerate(attributes):
        if best == best_attribute:
            for i, value in enumerate(unique_attributes(available_domains(deepcopy(data), attributes[j]))):
                if value == '?' and i == 0:
                    continue
                else:
                    subset = reduce_data(deepcopy(data), attributes[j], value)
                    if depth_limit is not None:
                        child = id3(subset, shrink_attributes(deepcopy(attributes), j), i, value, tree, depth_limit - 1)
                    else:
                        child = id3(subset, shrink_attributes(deepcopy(attributes), j), i, value, tree, depth_limit)
                    tree[value] = child
    return tree


# In[27]:


tree = id3(train, list(range(1, 23)), 1, 'p', {})


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