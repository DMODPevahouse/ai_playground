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