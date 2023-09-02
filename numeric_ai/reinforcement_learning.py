from operator import itemgetter
from copy import deepcopy
import random
import pprint

max_iterations = 10000
epsilon = .1


costs = { '.': -1, '*': -3, '^': -5, '~': -7}
costs


cardinal_moves = [(0,-1), (1,0), (0,1), (-1,0)]


reward = 10


def read_world(filename):
    result = []
    with open(filename) as f:
        for line in f.readlines():
            if len(line) > 0:
                result.append(list(line.strip()))
    return result


# <a id="create_value_world"></a>
# ## create_value_world
#
# The purpose of this function is to create an array of 0s that is the same size and same length as the world being used that will be adjusted to hold the state values **Used by**: [value_iteration](#value_iteration), [determinstic_value_iteration](#deterministic_value_iteration)
#
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
#
#
# **returns**: list[list]: this is the array that is desired, which is what is returned

# In[6]:


def create_value_world(world: list[list], goal: tuple, reward: int):
    result = []
    for y in range(len(world)):
        result.append([])
        for x in range(len(world[y])):
            result[y].append(0)
    x, y = goal
    result[y][x] = reward
    return result


# In[7]:


unit_world = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
unit_actions = [1, 1, 1]
unit_test = create_value_world(unit_world, (2, 2), 5)
assert unit_test == [[0, 0, 0], [0, 0, 0], [0, 0, 5]]
unit_test = create_value_world(unit_world, (1, 1), 10)
assert unit_test == [[0, 0, 0], [0, 10, 0], [0, 0, 0]]
unit_test = create_value_world(unit_world, (2, 0), 100)
assert unit_test == [[0, 0, 100], [0, 0, 0], [0, 0, 0]]


# <a id="deterministic_look_ahead"></a>
# ## deterministic_look_ahead
#
# This function was the original test function for looking ahead on what a specific action will result in a determinitistic value_iteration implementation. It took in the current state and location of the world to compare what the value would be a of a passed action **Used by**: [deterministic_value_iteration](#deterministic_value_iteration)
#
# * **action**: list[tuple]: The action being investigated
# * **costs**: dict: the costs of the terrain
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **x**: int: x value of the world location
# * **y**: int: y value of the world location
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# * **value_world**: the world that holds the current values for each location of the world
#
# **returns**: float: returns the total cost of the equation that will be used to figure out the next steps

# In[8]:


def deterministic_look_ahead(action: list[tuple], costs: dict, world: list[list], goal: tuple, reward: int, x: int, y: int, gamma: float, value_world: list[list]):
    goal_x, goal_y = goal
    if (y, x) == (goal_x, goal_y):
        return reward
    elif y+action[1] > (len(world[0]) - 1) or  y+action[1] < 0 or x+action[0] > (len(world) - 1) or  x+action[0] < 0:
        return -1000000
    elif world[x+action[0]][y+action[1]] == 'x' or world[x][y] == 'x':
        return -1000000
    else:
        current_cost = costs[world[x][y]]
        total = current_cost + (gamma * value_world[x+action[0]][y+action[1]])
        return total


# In[9]:


unit_world = read_world( "small.txt")
unit_value_world = create_value_world(unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5)
unit_test = deterministic_look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 0, 0, .9, unit_value_world)
assert unit_test == -1
unit_test = deterministic_look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, len(unit_world) - 1,len(unit_world[0]) - 1, .9, unit_value_world)
assert unit_test == 5
unit_test = deterministic_look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 1, 1, .9, unit_value_world)
assert unit_test == -3


# <a id="look_ahead"></a>
# ## look_ahead
#
# This function is to take the action that will be looked into to figure out which action would be the next best step to take for the algorithm to learn how to traverse the terrain **Used by**: [value_iteration](#value_iteration).
#
# * **action**: list[tuple]: The action being investigated
# * **costs**: dict: the costs of the terrain
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **x**: int: x value of the world location
# * **y**: int: y value of the world location
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# * **value_world**: the world that holds the current values for each location of the world
#
# **returns**: documentation of the returned value and type.

# In[10]:


def look_ahead(action: list[tuple], costs: dict, world: list[list], goal: tuple, reward: int, x: int, y: int, gamma: float, value_world: list[list]):
    goal_x, goal_y = goal
    if (y, x) == (goal_x, goal_y):
        return reward
    elif y+action[1] > (len(world[0]) - 1) or  y+action[1] < 0 or x+action[0] > (len(world) - 1) or  x+action[0] < 0:
        return -10000
    elif world[x+action[0]][y+action[1]] == 'x' or world[x][y] == 'x':
        return -10000
    else:
        total = (gamma * value_world[x+action[0]][y+action[1]])
        return total


# In[11]:


unit_world = read_world( "small.txt")
unit_value_world = create_value_world(unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5)
unit_test = look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 0, 0, .9, unit_value_world)
assert unit_test == 0.0
unit_test = look_ahead((0, 1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5,  len(unit_world) - 1, len(unit_world[0]) - 1, .9, unit_value_world)
assert unit_test == 5
unit_test = look_ahead((0, -1), costs, unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, 0, 0, .9, unit_value_world)
assert unit_test == -10000


# <a id="decide_action"></a>
# ## decide_action
#
# Here is where the actions that have been looked into will be decided, with safe guards to preventing the algorithm from going out of the terrain and using the random variable to decide which to select **Used by**: [select_action](#select_action)
#
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **action_list**: list: this contains, in order, a list of actions that have been investigated to see which is best
# * **x**: int: x value of the world location
# * **y**: int: y value of the world location
# * **costs**: dict: the costs of the terrain
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
#
#
# **returns**: tuple, float: returns a valid action and float from previous equations based off of state

# In[12]:


def decide_action(world: list[list], action_list: list, x: int, y: int, costs: dict, goal: tuple, reward: int):
    if (y, x) == goal:
        return (0, 0), reward
    if y+action_list[0][0][1] <= (len(world[0]) - 1) and y+action_list[0][0][1] >= 0 and x+action_list[0][0][0] <= (len(world) - 1) and x+action_list[0][0][0] >= 0 and world[x+action_list[0][0][0]][y+action_list[0][0][1]] != 'x':
        return action_list[0]
    elif y+action_list[1][0][1] <= (len(world[0]) - 1) and y+action_list[1][0][1] >= 0 and x+action_list[1][0][0] <= (len(world) - 1) and x+action_list[1][0][0] >= 0 and world[x+action_list[1][0][0]][y+action_list[1][0][1]] != 'x':
        return action_list[1]
    elif y+action_list[2][0][1] <= (len(world[0]) - 1) and y+action_list[2][0][1] >= 0 and x+action_list[2][0][0] <= (len(world) - 1) and x+action_list[2][0][0] >= 0 and world[x+action_list[2][0][0]][y+action_list[2][0][1]] != 'x':
        return action_list[2]
    else:
        return action_list[3]


# In[13]:


unit_world = read_world( "small.txt")
unit_action_list = [[(0 ,1), 1], [(1, 0), 2], [(-1, 0), 3], [(0, -1), 4]]
unit_test = decide_action(unit_world, unit_action_list, 0, 0, costs, (9, 9), 10)
assert unit_test == [(0, 1), 1]
unit_test = decide_action(unit_world, unit_action_list, 0, 0, costs, (9, 9), 10)
assert unit_test == [(0, 1), 1]
unit_test = decide_action(unit_world, unit_action_list, 1, 1, costs, (9, 9), 10)
assert unit_test == [(0, 1), 1]


# <a id="select_action"></a>
# ## select_action
#
# This is the function that will be called to gather all the information needed to take the actions that will be investigated, look into each specific action, then return the best valid action to the value iteration function **Uses**: [look_ahead](#look_ahead), [decide_action](#decide_action) **Used by**: [value_iteration](#value_iteration)
#
#
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **costs**: dict: the costs of the terrain
# * **actions**: list[tuple]: The actions being investigated
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **x**: int: x value of the world location
# * **y**: int: y value of the world location
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
# * **value_world**: the world that holds the current values for each location of the world
#
#
# **returns**: tuple, float: this returns the action that has been decided and the amount that was left with what was decided

# In[14]:


def select_action(world: list[list], costs: dict, actions: list[tuple], goal: tuple, reward: int, gamma: float, x: int, y: int, value_world: list[list]):
    action_list = []
    for main_action in actions:
        sum = 0
        if world[x][y] == 'x': sum = -10000
        else: sum = costs[world[x][y]] + look_ahead(main_action, costs, world, goal, reward, x, y, gamma, value_world)
        for action in actions:
            if action == main_action: continue
            else:
                q = look_ahead(action, costs, world, goal, reward, x, y, 1-gamma, value_world)
                sum = sum + q
                value_world[x][y] = sum
        action_list.append([main_action, sum])
    action_list = sorted(action_list, key=itemgetter(1), reverse=True)
    decided_action = decide_action(world, action_list, x, y, costs, goal, reward)
    return decided_action


# In[15]:


unit_world = read_world( "small.txt")
unit_value_world = create_value_world(unit_world, (len(unit_world[0]) - 1, len(unit_world) - 1), 5)
unit_test = select_action(unit_world, costs, cardinal_moves, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, .9, 0, 0, unit_value_world)
assert unit_test == [(1, 0), -20001.0]
unit_test = select_action(unit_world, costs, cardinal_moves, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, .9, 5, 5, unit_value_world)
assert unit_test == [(1, 0), -9996.5]
unit_test = select_action(unit_world, costs, cardinal_moves, (len(unit_world[0]) - 1, len(unit_world) - 1), 5, .9, 2, 2, unit_value_world)
assert unit_test == [(0, -1), -3.0]


# <a id="determinisitic_value_iteration"></a>
# ## determinisitic_value_iteration
#
# This was a test function to get a simplified version of deterministic value iteration working with low values to get an understanding of how value iteration works **Uses**: [deterministic_look_ahead](#deterministic_look_ahead), [create_value_world](#create_value_world)
#
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **costs**: dict: the costs of the terrain
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **actions**: list[tuple]: The actions being investigated
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
#
# **returns**: dict: the polict of every movement in the world

# In[16]:


def deterministic_value_iteration(world, costs, goal, rewards, actions, gamma):
    value_world, policy, delta = create_value_world(deepcopy(world), goal, rewards), {}, 0
    while True:
        delta += 1
        for y in range(len(world[0])):
            for x in range(len(world)):
                temp_action_costs = []
                for action in actions:
                    temp_action_costs.append([action, deterministic_look_ahead(action, costs, world, goal, rewards, x, y, gamma, value_world)])
                temp_action_costs = sorted(temp_action_costs, key=itemgetter(1), reverse=True)
                value_world[x][y] = temp_action_costs[0][1]
                policy[(x, y)] = temp_action_costs[0][0]
                if world[x][y] == 'x':
                    policy[(x, y)] = 'x'
        if delta > max_iterations: break
    return policy


# no unit test, as it is an alternative solution to the problem

# <a id="value_iteration"></a>
# ## value_iteration
#
# This is the main algorithm for value iteration, implemented in a schotastic fashion where it will take in the world state, costs, goal, reward, actions, and gamma to incorporate them into the desired mathmatical function with a small amount of randomness implemented in the equation to figure out which function will actually be called **Uses**: [select_action](#select_action), [create_value_world](#create_value_world)
#
# * **world**: list[list]: the world that is the basis for the creation of the array.
# * **costs**: dict: the costs of the terrain
# * **goal**: tuple: the location of the goal
# * **reward**: int: the reward of the goal
# * **actions**: list[tuple]: The actions being investigated
# * **gamma**: float: the value being multiplied to determine each reinforcement learning value.
#
# **returns**: dict: the polict of every movement in the world

# In[17]:


def value_iteration(world, costs, goal, rewards, actions, gamma):
    value_world, policy, delta = create_value_world(deepcopy(world), goal, rewards), {}, 0
    while True:
        delta += 1
        for y in range(len(world[0])):
            for x in range(len(world)):
                v_pre = value_world[x][y]
                next_action = select_action(world, costs, actions, goal, reward, gamma, x, y, value_world)
                value_world[x][y] = next_action[1]
                policy[(x, y)] = next_action[0]
                if world[x][y] == 'x':
                    policy[(x, y)] = 'x'
                v_after = value_world[x][y]
                if (y, x) == goal:
                    v_after = 10000
                    value_world[x][y] = rewards
        if delta > max_iterations: break
    return policy


# <a id="pretty_print_policy"></a>
# ## pretty_print_policy
#
# The purpose of this function is to take the policy of the value_iteration function and print it in an easy to read fashion
#
# * **cols**: list: amount of cols
# * **rows**: list: amount of rows
# * **policy**: dict: the values for the world for how to determine direction
# * **goal**: tuple: the location of the goal
#
# **returns**: documentation of the returned value and type.

# In[18]:


def pretty_print_policy( cols: list, rows: list, policy: dict, goal: tuple):
    directions = {(1, 0) : 'v', (0, 1) : '>', (-1, 0) : '^', (0, -1) : '<', 'x' : 'x', (0, 0) : 'H'}
    policy_printer = []
    for x in range(rows):
        row_printer = []
        for y in range(cols):
            if (y, x) == goal:
                 row_printer.append('H')
            else:
                row_printer.append(directions[policy[(x, y)]])
        policy_printer.append(row_printer)
    for item in policy_printer:
        print(*item, sep='')


# ## Value Iteration
#
# ### Small World

# In[19]:


small_world = read_world( "small.txt")


# In[20]:


goal = (len(small_world[0])-1, len(small_world)-1)
gamma = 0.9

small_policy = value_iteration(small_world, costs, goal, reward, cardinal_moves, gamma)


# In[21]:


cols = len(small_world[0])
rows = len(small_world)

#pretty_print_policy(cols, rows, test_policy, goal) I changed this to small_policy as that was what was ran top, hope that is ok
pretty_print_policy(cols, rows, small_policy, goal)


# ### Large World

# In[22]:


large_world = read_world( "large.txt")


# In[23]:


goal = (len(large_world[0])-1, len(large_world)-1) # Lower Right Corner FILL ME IN
gamma = 0.9

large_policy = value_iteration(large_world, costs, goal, reward, cardinal_moves, gamma)


# In[24]:


cols = len(large_world[0])
rows = len(large_world)

pretty_print_policy( cols, rows, large_policy, goal)