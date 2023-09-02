get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from copy import deepcopy


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


def view_sensor_image( data):
    figure = plt.figure(figsize=(4,4))
    axes = figure.add_subplot(1, 1, 1)
    pixels = np.array([255 - p * 255 for p in data[:-1]], dtype='uint8')
    pixels = pixels.reshape((4, 4))
    axes.set_title( "Left Camera:" + data[-1])
    axes.imshow(pixels, cmap='gray')
    plt.show()
    plt.close()



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


def learn_model(data, hidden_nodes, verbose=False):
    network = create_network(data, hidden_nodes)
    previous_error, current_error, counter = 0, 1, 0
    while abs(current_error - previous_error) > 0.00001:
        for line in data:
            calculated_hidden, calculated_output = calculate_output(deepcopy(line), deepcopy(network))
            delta_output = calculate_delta_output(network[1], calculated_output)
            delta_hidden = calculate_delta_hidden(network[1], calculated_hidden, delta_output)
            original_output = network[1]
            network = update_theta(network, data, delta_output, delta_hidden)
            previous_error, current_error = current_error, sum(
                calculated_output[i] - original_output[i] for i in list(range(len(calculated_output)))) / len(
                calculated_output)
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


def apply_model(model, test_data, labeled=False):
    # unfinished
    pass



