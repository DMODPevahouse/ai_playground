#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Module 13 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# # The Problem
# 
# When we last left our agent in Module 4, it was wandering around a world filled with plains, forests, swamps, hills and mountains. This presupposes a map with known terrain:
# 
# ```
# ......
# ...**.
# ...***
# ..^...
# ..~^..
# ```
# 
# but what if all we know is that we have some area of interest, that we've reduced to a GPS grid:
# 
# ```
# ??????
# ??????
# ??????
# ??????
# ??????
# ```
# 
# and the agent has to determine what kind of terrain is to the left, front and right of it?
# 
# Assuming the agent has a very simple visual sensor that constructs a 4x4 grayscale image for each of the three directions, it might it could see something like this:

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import random

plain =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,1.0, 1.0, 1.0, 1.0]
forest = [0.0, 1.0, 0.0, 0.0,1.0, 1.0, 1.0, 0.0,1.0, 1.0, 1.0, 1.0,0.0, 1.0, 0.0, 0.0]
hills =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0,0.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0]
swamp =  [0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,1.0, 0.0, 1.0, 0.0,1.0, 1.0, 1.0, 1.0]

figure = plt.figure(figsize=(20,6))

axes = figure.add_subplot(1, 3, 1)
pixels = np.array([255 - p * 255 for p in plain], dtype='uint8')
pixels = pixels.reshape((4, 4))
axes.set_title( "Left Camera")
axes.imshow(pixels, cmap='gray')

axes = figure.add_subplot(1, 3, 2)
pixels = np.array([255 - p * 255 for p in forest], dtype='uint8')
pixels = pixels.reshape((4, 4))
axes.set_title( "Front Camera")
axes.imshow(pixels, cmap='gray')

axes = figure.add_subplot(1, 3, 3)
pixels = np.array([255 - p * 255 for p in hills], dtype='uint8')
pixels = pixels.reshape((4, 4))
axes.set_title( "Right Camera")
axes.imshow(pixels, cmap='gray')

plt.show()
plt.close()


# which would be plains, forest and hills respectively.
# 

# ## The Assignment
# 
# In Assignment 12, we applied a logistic regression to determine if something was "hills" or "not hills". For this programming assignment your task is to write an artificial neural network that determines what kind of terrain it is. This is a multi-class problem.
# 
# For a starting point, you can refer to Pseudocode and the Self-Check.

# ## Data
# 
# As before, we have clean examples of the different types of terrain but based on the location, the registration can be a bit off for some of the types and the visual sensor is often blurry.
# 
# Here are the clean examples with different registrations: 

# In[3]:


clean_data = {
    "plains": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
    ],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
    ],
    "swamp": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]        
    ]
}


# Let's create a function that allows us to view any of these:

# In[4]:


def view_sensor_image( data):
    figure = plt.figure(figsize=(4,4))
    axes = figure.add_subplot(1, 1, 1)
    pixels = np.array([255 - p * 255 for p in data[:-1]], dtype='uint8')
    pixels = pixels.reshape((4, 4))
    axes.set_title( "Left Camera:" + data[-1])
    axes.imshow(pixels, cmap='gray')
    plt.show()
    plt.close()


# "I think that I shall never see a thing so lovely as a tree."

# In[5]:


view_sensor_image( clean_data[ "forest"][0])


# In[6]:


view_sensor_image( clean_data["swamp"][0])


# The data that comes in, however, is noisy. The values are never exactly 0 and 1. In order to mimic this we need a `blur` function.
# 
# We will assume that noise is normally distributed. For values that should be 0, the noisy values are distributed $N(0.10, 0.05)$. For values should be 1, the noisy values are distributed $N(0.9, 0.10)$.

# In[7]:


def blur( data):
    def apply_noise( value):
        if value < 0.5:
            v = random.gauss( 0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss( 0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v
    noisy_readings = [apply_noise( v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]


# We can see how this affects what the agent *actually* sees.

# In[8]:


view_sensor_image( blur( clean_data["swamp"][0]))


# You are going to want to write four (4) functions:
# 
# 1. `generate_data`
# 2. `learn_model`
# 3. `apply_model`
# 
# ### `generate_data`
# 
# With the clean examples and the `blur` function, we have an unlimited amount of data for training and testing our classifier, an ANN that determines if a sensor image is hills, swamp, forest or plains.
# 
# In classification, there is a general problem called the "unbalanced class problem". In general, we want our training data to have the same number of classes for each class. This means you should probably generate training data with, say, 100 of each type.
# 
# But what do we do about the class label with the neural network?
# 
# In this case, we can do "one hot". Instead of `generate_data` outputing a single 0 or 1, it should output a vector of 0's and 1's so that $y$ is now a vector as well as $x$. We can use the first position for hill, the second for swamp, the third for forest and the fourth for plains:
# 
# ```
# [0, 1, 0, 0]
# ```
# 
# what am I? swamp.
# 
# Unlike logistic regression, you should set the *biases* inside the neural network (the implict $x_0$ = 1) because there are going to be lot of them (one for every hidden and output node).
# 
# `generate_data` now only needs to take how many you want of each class:
# 
# `generate_data( clean_data, 100)`
# 
# generates 100 hills, 100 swamp, 100 forest, 100 plains and transforms $y$ into the respective "one hot" encoding. You can use the code from Module 12 as a starting point.
# 
# ### `learn_model`
# 
# `learn_model` is the function that takes in training data and actually learns the ANN. If you're up to it, you can implement a vectorized version using Numpy but you might start with the loopy version first.
# 
# *In the lecture, I mentioned that you usually should mean normalize your data but you don't need to do that in this case because the data is already on the range 0-1.*
# 
# You should add a parameter to indicate how many nodes the hidden layer should have.
# 
# When verbose is True, you should print out the error so you can see that it is getting smaller.
# 
# When developing your algorithm, you need to watch the error so you'll set verbose=True to start. You should print it out every iteration and make sure it is declining. You'll have to experiment with both epsilon and alpha; and it doesn't hurt to make alpha adaptive (if the error increases, make alpha = alpha / 10).
# 
# When you know that your algorithm is working, change your code so that the error is printed out only every 1,000 iterations (it takes a lot of iterations for this problem to converge, depending on your parameter values--start early).
# 
# `learn_model` returns the neural network. The hidden layer will be one vector of thetas for each hidden node. And the output layer will have its own thetas, one for each output (4 in this case). Return it as a Tuple: (List of List, List of List).
# 
# ### `apply_model`
# 
# `apply_model` takes the ANN (the model) and either labeled or unlabeled data. If the data is unlabeled, it will return predictions for each observation as a List of Tuples of the inferred value (0 or 1) and the actual probability (so something like (1, 0.73) or (0, 0.19) so you have [(0, 0.30), (1, 0.98), (0, 0.87), (0, 0.12)]. Note that unlike the logistic regression, the threshold for 1 is not 0.5 but which value is largest (0.98 in this case).
# 
# If the data is labeled, you will return a List of List of Tuples of the actual value (0 or 1) and the predicted value (0 or 1). For a single data point, you'll have the pairs of actual values [(0, 1), (0, 0), (0, 0), (1, 0)] is a misclassification and [(0, 0), (0, 0), (1, 1), (0, 0)] will be a correct classification. Then you have a List of *those*, one for each observation.
# 
# ###  simple evaluation
# 
# We have an "unlimited" supply of data so we'll just generate a training set and then a test set and see how well our neural network does. Use the error rate (incorrect classifications/total examples) for your evaluation metric. We'll learn about more sophisticated 
# 
# 1. generate training set (how many do you think you need?)
# 2. generate test set (how many is a good "test" of the network you built?)
# 3. loop over [2, 4, 8] hidden nodes:
#     1. train model and apply to train data, calculate error rate.
#     2. apply to test data and calculate error rate.
#     3. print error rate
#     
# Which number of hidden nodes did best?
# 
# **As always when working with Lists or Lists of Lists, be very careful when you are modifying these items in place that this is what you intend (you may want to make a copy first)**

# In[9]:


import math
from copy import deepcopy


# <a id="create_network"></a>
# ## create_network
# 
# This function takes in the data, and the amount of hidden nodes and creates a network based on the length of lines of data  **Used by**: [learn_model](#learn_model)
# 
# * **data**: list[list]: the list of the data that the network will be built off of
# * **hidden_nodes**: int: the amount of hidden nodes there will be
# 
# **returns**: list[list]: this function will return a list of list with the first value being the hidden nodes, and the second being the output

# In[10]:


def create_network(data: list[list], hidden_nodes: int):
    network, hidden, output = [], [], []
    for i in list(range(hidden_nodes)):
        hidden_layer = []
        for j in list(range(len(data[0]) - 1)):
            hidden_layer.append(random.random())
        hidden.append(hidden_layer)
    network.append(hidden)
    for i in list(range(len(data[0]) - 1)):
        output.append(random.random())
    network.append(output)
    return network


# In[11]:


def create_network_experimental(data: list[list], hidden_nodes: int):
    network, hidden, output = [], [], []
    for i in list(range(hidden_nodes)):
        hidden_layer = []
        for j in list(range(len(data[0]) - 1)):
            hidden_layer.append(random.random())
        hidden.append(hidden_layer)
    network.append(hidden)
    for key in clean_data:
        output_list = []
        for i in list(range(len(data[0]) - 1)):
            output_list.append(random.random())
        output.append(output_list)
    network.append(output)
    return network


# In[12]:


unit_data = [blur(clean_data['forest'][0]), blur(clean_data['swamp'][0]), blur(clean_data['hills'][0])]
unit_test = create_network(unit_data, 2)
assert len(unit_test[0]) == 2
assert len(unit_test[1]) == 16
unit_test = create_network(unit_data, 4)
assert len(unit_test[0]) == 4
assert len(unit_test[1]) == 16
unit_test = create_network(unit_data, 8)
assert len(unit_test[0]) == 8
assert len(unit_test[1]) == 16


# <a id="sigmoid"></a>
# ## sigmoid
# 
# Simple function thats purpose is to return the answer of the sigmoid function and recieves a z value  **Used by**: [calculate_layer_output](#calculate_layer_output)
# 
# * **sum**: float: the appropriate value for the z value for the equation.
# 
# **returns**: float: the value of the sigmoid function

# In[13]:


def sigmoid(sum: float):
    return 1 / (1 + (math.e ** -sum))


# In[14]:


unit_test = sigmoid(5)
assert unit_test == 0.9933071490757153
unit_test = sigmoid((.5 * .2 + .16 * .5))
assert unit_test == 0.5448788923735801
unit_test = sigmoid(.5)
assert unit_test == 0.6224593312018546


# <a id="dot_product"></a>
# ## dot_product
# 
# This function takes in two lists that need to be multiplied together using the dot product.  **Used by**: [calculate_layer_output](#calculate_layer_output)
# 
# * **xs**: one vertex needed for a dot product
# * **ys**: one vertex needed for a dot product
# 
# **returns**: float: returns the dot product of the two lists

# In[15]:


def dot_product(xs: list, ys: list):
    return sum(x * y for x, y in zip(xs, ys))


# In[16]:


unit_xs, unit_ys = [1, 2, 3], [3, 2, 1]
unit_test = dot_product(unit_xs, unit_ys)
assert unit_test == 10
unit_xs, unit_ys = [10, 2, 30], [3, 20, 1]
unit_test = dot_product(unit_xs, unit_ys)
assert unit_test == 100
unit_xs, unit_ys = [.5, .2, .123], [.435, .23, .44]
unit_test = dot_product(unit_xs, unit_ys)
assert unit_test == 0.31762


# <a id="calculate_layer_output"></a>
# ## calculate_layer_output
# 
# The purpose of this function is to determine what the value of calculating the layers, whether it is the output layer or the hidden layer **Uses**: [sigmoid](#sigmoid), [dot_product](#dot_product) **Used by**: [calculate_output](#calculate_output)
# 
# * **input**: list: the current data being evaluated, as could be from the hidden layer here
# * **layer**: list[list]: this is the layer to be used, either the hidden or output both will be accounted for
# 
# **returns**: list: returns the output of the calculations using dot product and sigmoid will bring to determine thetas and errors of the network to see how it will learn

# In[17]:


def calculate_layer_output_backup(input: list, layer: list): #kept as a backup during experimentation
    output = []
    if isinstance(layer[0], list):
        output_list = [[] for _ in range(len(layer))]
        for i, node in enumerate(layer):
            output_list[i].append(sigmoid(dot_product(input, node)))
        output = output_list 
    elif not isinstance(layer[0], list):
        output.append(sigmoid(dot_product(input, layer)))
    return output


# In[18]:


def calculate_layer_output(input: list, layer: list):
    output = []
    if isinstance(layer[0], list):
        for node in layer:
            output.append(sigmoid(dot_product(input, node)))
    elif not isinstance(layer[0], list):
        output.append(sigmoid(dot_product(input, layer)))
    return output


# In[19]:


unit_layer = [1, 1, 2, 3]
unit_input = [3, 4, 3, 5]
unit_test = calculate_layer_output(unit_input, unit_layer)
assert unit_test == [0.9999999999993086]
unit_layer = [.1, .1, .2, .3]
unit_input = [.3, .4, .3, .5]
unit_test = calculate_layer_output(unit_input, unit_layer)
assert unit_test == [0.569546223939229]
unit_layer = [[.16, .45, .52, .63], [.1, .1, .2, .3], [1, 1, 2, 3]]
unit_input = [.8, .2, .13, .75]
unit_test = calculate_layer_output(unit_input, unit_layer)
assert unit_test == [0.6809410811604273, 0.5868600552894299, 0.9709709641636592]


# <a id="calculate_output"></a>
# ## calculate_output
# 
# This function takes in the network and current line being evaluated in order to determine how it will be used to change and teach the network **Uses**: [calculate_layer_output](#calculate_layer_output) **Used by**: [learn_model](#learn_model)
# 
# * **data_line**: list: the current data being learned
# * **network**: list[list]: this is the neural network that is being taught
# 
# **returns**: list, list: returns the values of the output for both the hidden layer and output layer

# In[20]:


def calculate_output(data_line: list[list], network: list[list]):
    hidden_layer = []
    data_line = data_line + [1.0]
    hidden_layer = calculate_layer_output(data_line, network[0])
    output_layer = calculate_layer_output(hidden_layer, network[1])
    return hidden_layer, output_layer


# In[21]:


unit_data = [[.5, .4, .3, .4], [.1, .4, .3, .4], [.1, .2, .7, .5]]
unit_network = [[[1, 1, 2, 3], [1, 3, 4, 6]], [.1, .2, .5, .6]]
unit_test1, unit_test2 = calculate_output(unit_data[0], unit_network)
assert unit_test1 == [0.9370266439430035, 0.995033198349943]
assert unit_test2 == [0.5726592856641474]
unit_test1, unit_test2 = calculate_output(unit_data[1], unit_network)
assert unit_test1 == [0.9088770389851438, 0.9926084586557181]
assert unit_test2 == [0.5718515370589238]
unit_test1, unit_test2 = calculate_output(unit_data[2], unit_network)
assert unit_test1 == [0.9608342772032357, 0.998498817743263] 
assert unit_test2 == [0.5734113607064428]


# <a id="calculate_delta_output"></a>
# ## calculate_delta_output
# 
# this calculates the delta of the output that will be use to update the thetas **Used by**: [learn_model](#learn_model).
# 
# * **original**: list: the original output
# * **calculated_output**: list: this is the output that was calculated using the network and new line of data.
# 
# **returns**: list: returns a list of floats that contain the new theta values

# In[22]:


def calculate_delta_output(original: list, calculated_output: list):
    delta_output = []
    for i in list(range(len(original))):
        delta_output.append(original[i] * (1 - calculated_output[0]) * (original[i] - calculated_output[0]))
    return delta_output


# In[23]:


unit_original = [1, 2, 5, 9]
unit_output = [2]
unit_test = calculate_delta_output(unit_original, unit_output)
assert unit_test == [1, 0, -15, -63]
unit_original = [.1, .2, .5, .9]
unit_output = [.2]
unit_test = calculate_delta_output(unit_original, unit_output)
assert unit_test == [-0.008000000000000002, 0.0, 0.12, 0.504]
unit_original = [.1, .2, .5, .9]
unit_output = [.32]
unit_test = calculate_delta_output(unit_original, unit_output)
assert unit_test == [-0.014959999999999998, -0.016319999999999998, 0.06119999999999999, 0.35496000000000005]


# <a id="calculate_delta_hidden"></a>
# ## calculate_delta_hidden
# 
# this function is used to follow the error equation to determine the deltas in the network **Used by**: [learn_model](#learn_model)
# 
# * **theta_output**: list: this is the thetas found by the calculating with the output
# * **calculated_hidden**: list: this is the calculation of the hidden nodes with the line of data
# * **delta_output**: list: this is the delta of the output list
# 
# **returns**: list: this function returns a list of deltas found with the hidden layer of the network which will be used for the neural network to learn

# In[24]:


def calculate_delta_hidden(theta_output: list, calculated_hidden: list, delta_output: list):
    delta_hidden = []
    for i, value in enumerate(calculated_hidden):
        delta_hidden.append(calculated_hidden[i] * (1 - calculated_hidden[i]) * sum(delta_output[j] * theta_output[j] for j in list(range(len(theta_output)))))
    return delta_hidden


# In[25]:


unit_original = [.1, .2, .5, .9]
unit_hidden = [.2, .2, .2, .2]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_test = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
assert unit_test == [0.08204800000000002,
 0.08204800000000002,
 0.08204800000000002,
 0.08204800000000002]
unit_original = [.1, .2, .5, .9]
unit_hidden = [.32, .12, .62, .42]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_test = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
assert unit_test == [0.0751381504, 0.036464102400000004, 0.08135362240000002, 0.08411605440000001]
unit_original = [.31, .42, .15, .79]
unit_hidden = [.32, .12, .62, .42]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_test = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
assert unit_test == [0.04530499020800001, 0.021986245248000004, 0.049052645648000014, 0.050718270288000016]


# <a id="update_layer_theta"></a>
# ## update_layer_theta
# 
# This function takes in the specific layer, hidden or output, the delta for that layer, and the current input to be used to update the specific layer in the network **Used by**: [update_theta](#update_theta)
# 
# * **layer**: list: the current layer being investigated to determine how it should be updated
# * **delta**: list: the delta to be investigated on layer.
# * **input**: list: the current data line being evaluated
# * **learning_rate**: float: this is the rate at which is defualted to .1.
# 
# **returns**: documentation of the returned value and type.

# In[26]:


def update_layer_theta(layer: list, delta: list, input:list, learning_rate:float=.1):
    updated_layer = []
    for i, node in enumerate(layer):
        updated_node = [[] for _ in range(len(delta))]
        if isinstance(node, list):
            for j, actual_value in enumerate(node):
                for k, delt in enumerate(delta):
                    updated_node[k].append(layer[i][j] + learning_rate * delt * input[i][j])
            updated_layer = updated_node
        elif not isinstance(node, list):
            sum = 0
            for hidden in input:
                for value in hidden:
                    sum = sum + value 
            updated_layer.append(layer[i] + learning_rate * delta[0] * sum)
    return updated_layer


# In[27]:


unit_data = [[.5, .4, .3, .4], [.1, .4, .3, .4], [.1, .2, .7, .5]]
unit_network = [[[1, 1, 2, 3], [1, 3, 4, 6]], [.1, .2, .5, .6]]
unit_original = [.1, .2, .5, .9]
unit_hidden = [.2, .2, .2, .2]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_delta = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
unit_test = update_layer_theta(unit_network[0], unit_delta, unit_data)
assert unit_test == [[1.00082048, 3.00328192, 4.00246144, 6.00328192], [1.00082048, 3.00328192, 4.00246144, 6.00328192], [1.00082048, 3.00328192, 4.00246144, 6.00328192], [1.00082048, 3.00328192, 4.00246144, 6.00328192]]
unit_test2 = update_layer_theta(unit_network[1], unit_delta, unit_network[0])
assert unit_test2 == [0.27230080000000007, 0.3723008000000001, 0.6723008, 0.7723008]
unit_test2 = update_layer_theta(unit_network[1], unit_delta, unit_test)
assert unit_test2 == [0.5597919299665922, 0.6597919299665922, 0.9597919299665922, 1.059791929966592]


# <a id="update_theta"></a>
# ## update_theta
# 
# The purpose of this function is to taken in all of the deltas as well as the current input and network in order to adjust the thetas based off of the deltas to continue progressing the neural network **Uses**: [update_layer_theta](#update_layer_theta)**Used by**: [learn_mode](#learn_model)
# 
# * **network**: list: the current network being investigated to determine how it should be updated
# * **input**: list: the current data line being evaluated
# * **delta_output**: list: the delta to be investigated on output layer.
# * **delta_hidden**: list: the delta to be investigated on hidden layer.
# * **learning_rate**: float: this is the rate at which is defualted to .1.
# 
# **returns**: list[list]: returns the adjusted network

# In[28]:


def update_theta(network: list, input: list, delta_output: list, delta_hidden:list, learning_rate: float=0.1):
    network[0] = update_layer_theta(network[0], delta_hidden, input, learning_rate)
    network[1] = update_layer_theta(network[1], delta_output, network[0], learning_rate)
    return network


# In[29]:


unit_data = [[.5, .4, .3, .4], [.1, .4, .3, .4], [.1, .2, .7, .5]]
unit_network = [[[1, 1, 2, 3], [1, 3, 4, 6]], [.1, .2, .5, .6]]
unit_original = [.1, .2, .5, .9]
unit_hidden = [.2, .2, .2, .2]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_delta = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
unit_test = update_theta(unit_network, unit_data, unit_delta, unit_hidden)
assert unit_test == [[[1.002, 3.008, 4.006, 6.008], [1.002, 3.008, 4.006, 6.008], [1.002, 3.008, 4.006, 6.008], [1.002, 3.008, 4.006, 6.008]], [0.5602564608000002, 0.6602564608000003, 0.9602564608000003, 1.0602564608000002]]
unit_data = [[.5, .4, .3, .4], [.1, .4, .3, .4], [.1, .2, .7, .5]]
unit_network = [[[.1, .1, .2, .3], [.1, .3, .4, 6]], [.1, .2, .5, .6]]
unit_original = [.1, .2, .5, .9]
unit_hidden = [.2, .2, .2, .2]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_delta = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
unit_test = update_theta(unit_network, unit_data, unit_delta, unit_hidden)
assert unit_test == [[[0.10200000000000001, 0.308, 0.406, 6.008], [0.10200000000000001, 0.308, 0.406, 6.008], [0.10200000000000001, 0.308, 0.406, 6.008], [0.10200000000000001, 0.308, 0.406, 6.008]], [0.3239582208000001, 0.4239582208000001, 0.7239582208, 0.8239582208]]
unit_data = [[.5, .4, .3, .4], [.1, .4, .3, .4], [.1, .2, .7, .5]]
unit_network = [[[.31, .11, .22, .35], [.15, .23, .14, .66]], [.21, .42, .35, .26]]
unit_original = [.1, .2, .5, .9]
unit_hidden = [.2, .2, .2, .2]
unit_output = calculate_delta_output(unit_original, unit_hidden)
unit_delta = calculate_delta_hidden(unit_original, unit_hidden, unit_output)
unit_test = update_theta(unit_network, unit_data, unit_delta, unit_hidden)
assert unit_test == [[[0.152, 0.23800000000000002, 0.14600000000000002, 0.668], [0.152, 0.23800000000000002, 0.14600000000000002, 0.668], [0.152, 0.23800000000000002, 0.14600000000000002, 0.668], [0.152, 0.23800000000000002, 0.14600000000000002, 0.668]], [0.24951431680000002, 0.4595143168, 0.3895143168, 0.2995143168]]


# ---
# 
# Put your helper functions above here.
# 
# ## Main Functions

# ### generate_data
# 
# Generates an endless supply of blurred data from a collection of terrain prototypes.
# 
# * `data`: Dict[Str, List[Any]] - a Dictionary of "clean" prototypes for each landscape type.
# * `n`: Int - the number of blurred examples of each terrain type to return.
# 
# returns
# 
# * List[List[Any]] - a List of Lists. Each individual List is a blurred example of a terrain type, generated from the prototype.

# In[30]:


def generate_data(data: list[list], n: int):
    generated_data = []
    for i in list(range(n)):
        for key in data:
            generated_data.append(blur(data[key][0]))
    return generated_data
    


# <a id="learn_model"></a>
# ## learn_model
# 
# Here resides the main function that will be used by taking in the data, amount of hidden nodes, and a optional flag for more information that will be used to create the network to apply classification for the agent to discover the surroundings **Uses**: [calculate_output](#calculate_output), [create_network](#create_network), [calculate_delta_output](#calculate_delta_output), [calculate_delta_hidden](#calculate_delta_hidden), [update_theta](#update_theta)
# 
# * **data**: list[list]: 
# * **hidden_nodes**: int: used to determine the amount of hidden nodes to be used
# * **verbose**: bool: used to determine if there will be additional information shown or not
# 
# **returns**: tuple: returns the tuple of the hidden nodes and the output

# In[31]:


def learn_model( data, hidden_nodes, verbose=False):
    network = create_network(data, hidden_nodes)
    previous_error, current_error, counter = 0, 1, 0
    while abs(current_error - previous_error) > 0.00001:
        for line in data:
            calculated_hidden, calculated_output = calculate_output(deepcopy(line), deepcopy(network))
            delta_output = calculate_delta_output(network[1], calculated_output)
            delta_hidden = calculate_delta_hidden(network[1], calculated_hidden, delta_output)
            original_output = network[1]
            network = update_theta(network, data, delta_output, delta_hidden)
            previous_error, current_error = current_error, sum(calculated_output[i] - original_output[i] for i in list(range(len(calculated_output)))) / len(calculated_output)
            if verbose and counter == 1000:
                print("error is:", abs(current_error - previous_error))
                counter = 0
            counter += 1
    return (network[0], network[1])


# <a id="apply_model"></a>
# ## apply_model
# 
# This function was never finished but should it have been its purpose would be to take in the model that was built by learn model, take in a set of test data, and apply classifications to that test data
# 
# * **model**: tuple: the model that was built and tuned for testing 
# * **test_data**: list[list]: the data seeking a classification
# * **verbose**: bool: this optional variable is for if the test data is labeled or not 
# 
# **returns**: tuple: returns the tuple of the hidden nodes and the output

# In[33]:


def apply_model( model, test_data, labeled=False):
    # there was a fundamental misunderstanding of what I needed to do with the learn_model, leaving me in a position 
    # where I was not going to get a valid classification. Further explanation below
    pass


# Test out generate_data:

# In[34]:


results = generate_data( clean_data, 10)
for result in results:
    print( result)


# Use `generate_data` to generate 100 blurred examples of each type (all four terrains).

# In[35]:


train_data = generate_data( clean_data, 100)


# Use `learn_model` to learn a ANN model for classifying sensor images as hills, swamps, plains or forest. **Set Verbose to True**

# In[36]:


model = learn_model( train_data, 2, True)


# Use `generate_data` to generate 100 blurred examples of each terrain and use this as your test data.

# In[37]:


test_data = generate_data( clean_data, 100)


# Apply the model and evaluate the results.

# In[ ]:


results = apply_model( model, test_data)


# In[ ]:


print( results)


# Now that you're pretty sure your algorithm works (the error rate during training is going down, and you can evaluate `apply_model` results for its error rate), you need to determine what the best number of hidden nodes is.
# 
# Try 2, 4, or 8 hidden nodes and indicate the best one. Follow the outline above under "Simple Evaluation".
# In the "real world", you could 10 fold cross validation and validation curves to determine the number of hidden nodes and possibly if you needed one or two hidden layers.

# ## Struggles
# I unfortunately did not get to the cross validation. I was prepared to do it after I figured out my classification function, apply_model, but I realized I had drastically built my model wrong. More explanation on the bottom.

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

# In[38]:


def error_rate(y: list, y_hat: list, total: int):
    error_rates = 0
    for i, correct in enumerate(y):
        if y_hat[i] != correct:
            error_rates += 1
    metric = error_rates/total
    return metric


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

# In[39]:


def evaluate(data: list[list], model: list, evaluation_fn, classify_fn):
    y_hats, ys = [], []
    for row in data:
        ys.append(row[0])
    y_hats_results = classify_fn(model, data)
    for row in y_hats_results:
        y_hats.append(row[0])
    metric = evaluation_fn(ys, y_hats, len(y_hats))
    return metric


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

# In[40]:


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


# In[424]:


# no tests were done due to my classification not working


# which number of hidden nodes is best? ____

# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# ## Explanation on progress
# 
# Sorry for turning in an uncompleteded assignment. The main reason for this was I misread the instructions. Somehow I got it in my head after reading the instructions that the output needed to include a value for each line in the data, so my output had 16 values based on that.
# 
# I spent 22 hours getting that working after struggling with understanding how the neural networks would be built and trained. Once I got that working, I thought classification would be easy. Then I printed what the finished model looked like and it dinged in my brain that it did not make a lot of sense. After thinking about how to shape that data into something that could be classified I deemed that it was not truly feasable to build any worth while classifications from that data. 
# 
# That unfortunately means that after dozens of hours of work, most of what I did was wasted and would need to be refactored to follow the instructions that once I thought about how to apply the model and reread the instructions I had missed. 
# 
# I spent a couple dozen hours all together on progressing where I am now and do not think I would have time to refactor/rebuild an applicable solution based on the instructions, so I wanted to make sure I got something turned in to show the effort I put into it. Should time free up tomorrow and I can refactor my current solution, I will make another submission before the due date, otherwise this is where I got too.
# 
# A few specific areas I struggled with getting to where I did get, and why it took so long to get there, is getting all of the mathmatical equations lined up, where to use them, and how to follow along with them. I struggled getting that down and realized to late that you updated office hours for neural networks. I had only noticed for a week 14 office hours and not this one. So that is on me as well
# 
# I apologize for the waste of time on this assignment and thank you for the semester. Have a great rest of your summer. Below is proof that it would run at least with more hidden nodes, even though it was not correct.

# In[44]:


model_4 = learn_model( train_data, 4)


# In[45]:


model_4


# In[46]:


model_8 = learn_model( train_data, 8)


# In[47]:


model_8


# In[ ]:




