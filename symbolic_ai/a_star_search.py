from typing import List, Tuple, Dict, Callable
from copy import deepcopy


COSTS = { '🌾': 1, '🌲': 3, '🪨': 5, '🐊': 7}

MOVES = [(0,-1), (1,0), (0,1), (-1,0)]

small_world = [
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾']
]

full_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🗻', '🗻', '🗻', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾'],
['🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾'],
['🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌾', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🐊', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌲', '🌲', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🪨', '🪨', '🪨', '🪨', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨', '🪨', '🌾', '🐊', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🌾', '🌾', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🪨', '🪨', '🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🗻', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🪨', '🪨', '🗻', '🗻', '🌾', '🗻', '🗻', '🪨', '🪨', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🪨', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾']
]


# <a id="heuristic"></a>
# ## heuristic
#
# This function creates h(n), which is an estimating value of the shortest distance that it would take to reach the goal, but it can never surpass the answer, otherwise it is inadmissable. The logic here is to take Manhattan distance at the cheapest cost (plains) as it would be impossible to get lower then that. **Used by** [assign_g_h](#assign_g_h)
#
# * **location** Tuple[int, int]: the current location of the bot, `(x, y)`.
# * **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
#
# **returns** int - the value of the heuristic to be used

# In[6]:


def heuristic(location: Tuple[int, int], goal: Tuple[int, int]):
    if goal[0] > location[0]:
        a = goal[0] - location[0]
    else:
        a = location[0] - goal[0]
    if goal[1] > location[1]:
        b = goal[1] - location[1]
    else:
        b = location[1] - goal[1]
    return a + b


# In[7]:


goal = (8, 8)
location = (0, 0)
actual = heuristic(location, goal)
assert actual == 16
goal = (0, 0)
location = (5, 5)
actual = heuristic(location, goal)
assert actual == 10
goal = (5, 1)
location = (3, 7)
actual = heuristic(location, goal)
assert actual == 8


# <a id="assign_heuristics"></a>
# ## assign_g_h
#
# This function is used to assign the cost of g, and the value of h to the a star function to evaluate the approximate cost needed for f(n) = g(n) + h(n) **Uses**: [heuristic](#heuristic). **Used by** [a_star_search](#a_star_search): (links to functions used by).
#
# * **world** List[List[str]]: the actual context for the navigation problem.
# * **location** Tuple[int, int]: the current location of the bot, `(x, y)`.
# * **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
#
# **returns**: int or str. Int if it is capable of being traveled, string if it is `impassible`

# In[8]:


def assign_g_h(world: List[List[str]], location: Tuple[int, int], goal: Tuple[int, int]):
    a = location[0]
    b = location[1]
    if world[a][b] == '🗻':
        g = 'impassible'
        return g
    else:
        g = COSTS[world[a][b]]
    h = heuristic(location, goal)
    return g + h


# In[9]:


f = assign_g_h(full_world, (0, 0), (5, 5))
assert f == 11
f = assign_g_h(full_world, (3, 4), (8, 1))
assert f == 13
f = assign_g_h(full_world, (4, 5), (0, 7))
assert f == 'impassible'


# <a id="move_actions"></a>
# ## move_actions
#
# The purpose of this function is to determine all of the possible travel locations that can be reached by the current location in the path, then sends back the new location, how it traveled, and the starting location **Uses**: None **Used by**: [a_star_search](#a_star_search).
#
# * **world** List[List[str]]: the actual context for the navigation problem.
# * **location** Tuple[int, int]: the current location of the bot, `(x, y)`.
# * **movement** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
#
#
# **returns**: List[Tuple[int, int]] - this function returns a list of the new location, type of movement, and previous location

# In[10]:


def move_actions(world: List[List[str]], location: Tuple[int, int], movement: List[Tuple[int, int]]):
    successors = []
    left = location[1] - movement[0][1]
    right = location[1] - movement[2][1]
    up = location[0] - movement[3][0]
    down = location[0] - movement[1][0]
    if left >= 0 and left < len(world[0]):
        successors.append(((location[0], left), movement[1], location))
    if right >= 0 and right < len(world[0]):
        successors.append(((location[0], right), movement[3], location))
    if up >= 0 and up < len(world):
        successors.append(((up, location[1]), movement[2], location))
    if down >= 0 and down < len(world):
        successors.append(((down, location[1]), movement[0], location))
    return successors


# In[11]:


start = (0, 0)
test = move_actions(full_world, start, MOVES)
assert test == [((0, 1), (1, 0), (0, 0)), ((1, 0), (0, 1), (0, 0))]
start = (1, 1)
test = move_actions(full_world, start, MOVES)
assert test == [((1, 2), (1, 0), (1, 1)), ((1, 0), (-1, 0), (1, 1)), ((2, 1), (0, 1), (1, 1)),
                ((0, 1), (0, -1), (1, 1))]
start = (len(full_world[0]) - 1, len(full_world) - 1)
test = move_actions(full_world, start, MOVES)
assert test == [((26, 25), (-1, 0), (26, 26)), ((25, 26), (0, -1), (26, 26))]


# <a id="discover_path"></a>
# ## discover_path
#
# This function takes in the explored list from a_star_search and uses the third variable to backtrack in the explored list to find the actual path that is the quickest to use. Without this function, a_star_search would just find the goal, not find the path to the goal **Uses**: none. **Used by**: [a_star_search](#a_star_search).
#
# * **explored**: all the locations that have been explored to reach the goal
# * **start**: The starting point of the bot, '(x, y)'
#
# **returns**: Returns List[Tuple[int, int]]: the offsets needed to get from start state to the goal as a `List`.

# In[12]:


def discover_path(explored: List[List[Tuple[int, int]]], start: Tuple[int, int]):
    pathfinding = explored[-1]
    path = [pathfinding[1]]
    while pathfinding[2] is not start:
        for x in explored:
            if x[0] == pathfinding[2]:
                pathfinding = x
        path.insert(0, pathfinding[1])
    return path


# In[13]:


path = discover_path([[(0, 1), (0, 1), (0, 0)], [(0, 2), (0, 1), (0, 1)], [(0, 3), (0, 1), (0, 2)]], (0, 0))
assert path == [(0, 1), (0, 1), (0, 1)]
path = discover_path([[(1, 0), (1, 0), (0, 0)], [(2, 0), (1, 0), (1, 0)], [(3, 0), (1, 0), (2, 0)]], (0, 0))
assert path == [(1, 0), (1, 0), (1, 0)]
path = discover_path([[(6, 5), (-1, 0), (5, 5)], [(6, 6), (0, -1), (6, 5)], [(6, 7), (0, -1), (6, 6)]], (5, 5))
assert path == [(-1, 0), (0, -1), (0, -1)]


# *add as many markdown and code cells here as you need for helper functions. We have added `heuristic` for you*

# <a id="a_star_search"></a>
# ## a_star_search
#
# The key to a_star_search is it is similar to a greedy best first search. The way it is an improvement is it calls the function heuristic to establish an estimated cost based on this cost of the movement to a location plus an estimated cost of that location to the goal. The heuristic used here was Manhattan distance based on a cost of 1. After that it find the cheapest path from start to goal continually updating that general cost of the path in the world. **USES** [move_actions](#move_actions), [discover_path](#discover_path)
#
# * **world** List[List[str]]: the actual context for the navigation problem.
# * **start** Tuple[int, int]: the starting location of the bot, `(x, y)`.
# * **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
# * **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
# * **moves** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
# * **heuristic** Callable: is a heuristic function, $h(n)$.
#
#
# **returns** List[Tuple[int, int]]: the offsets needed to get from start state to the goal as a `List`.
#

# In[14]:


def a_star_search(world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int],
                  moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    explored = []
    h = 0
    explored_coordinates = []
    frontier = [[h, start, (0, 0), (0, 0)]]
    while frontier:
        frontier.sort(key=lambda x: x[0])
        first_frontier = frontier.pop(0)
        explored.append((first_frontier[1], first_frontier[2], first_frontier[3]))
        explored_coordinates.append(first_frontier[1])
        if first_frontier[1] == goal:
            path = discover_path(explored, start)
            return path
        for successor in move_actions(world, first_frontier[1], moves):
            if successor[0] not in explored_coordinates:
                f = assign_g_h(world, successor[0], goal)
                if f != 'impassible':
                    f = f + h
                    frontier.append((f, successor[0], successor[1], successor[2]))
    return None


# In[15]:


test = a_star_search(small_world, (0, 0), (3, 3), COSTS, MOVES, heuristic)
assert test == [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0)]
test = a_star_search(full_world, (0, 0), (3, 3), COSTS, MOVES, heuristic)
assert test == [(1, 0), (1, 0), (1, 0), (0, 1), (0, 1), (0, 1)]
small_start = (0, 0)
small_goal = (len(small_world[0]) - 1, len(small_world) - 1)
test = a_star_search(small_world, small_start, small_goal, COSTS, MOVES, heuristic)
assert test == [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1), (0, 1), (0, 1)]


# <a id="pretty_print_path"></a>
# ## pretty_print_path
#
# This function takes in the value of the world, start, and goal then prints out the path from start to goal in a manner that is pleasing to the eyes. It uses a dict to determine the path, and before hand uses the global COSTS to return the entire cost of the path. This function is called explicitly and uses no functions, only deepcopy
#
# * **world** List[List[str]]: the world (terrain map) for the path to be printed upon.
# * **path** List[Tuple[int, int]]: the path from start to goal, in offsets.
# * **start** Tuple[int, int]: the starting location for the path.
# * **goal** Tuple[int, int]: the goal location for the path.
# * **costs** Dict[str, int]: the costs for each action.
#
# **returns** int - The path cost.

#
# <div style="background: lemonchiffon; margin:20px; padding: 20px;">
#     <strong>Important</strong>
#     <p>
#         Does your output of pretty_print_path really look like the specification? Go check again.
#     </p>
# </div>

# In[16]:


def pretty_print_path(world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int],
                      goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    directions = {(0, 1): '⏬', (1, 0): '⏩', (0, -1): '⏫', (-1, 0): '⏪'}
    a = start[0]
    b = start[1]
    a, b = start[0], start[1]
    new_world = deepcopy(world)
    new_world[goal[0]][goal[1]] = '🎁'
    the_expense = costs[world[a][b]]
    for travel in path:
        if a < len(new_world[0]) and b < len(new_world):
            the_expense = the_expense + costs[world[a][b]]
            new_world[a][b] = directions[travel]
        if travel[1] != 0:
            a = a + travel[1]
        else:
            b = b + travel[0]
    for item in new_world:
        print(*item, sep='')
    return the_expense


# In[17]:


test_path = a_star_search(full_world, (1, 1), (0, 0), COSTS, MOVES, heuristic)
unit_test = pretty_print_path(full_world, test_path, (1, 1), (0, 0), COSTS)
assert unit_test == 3
test_path = a_star_search(full_world, (8, 8), (0, 0), COSTS, MOVES, heuristic)
unit_test = pretty_print_path(full_world, test_path, (8, 8), (0, 0), COSTS)
assert unit_test == 29
test_path = a_star_search(full_world, (9, 19), (0, 0), COSTS, MOVES, heuristic)
unit_test = pretty_print_path(full_world, test_path, (9, 19), (0, 0), COSTS)
assert unit_test == 59

small_start = (0, 0)
small_goal = (len(small_world[0]) - 1, len(small_world) - 1)
small_path = a_star_search(small_world, small_start, small_goal, COSTS, MOVES, heuristic)
small_path_cost = pretty_print_path(small_world, small_path, small_start, small_goal, COSTS)
print(f"total path cost: {small_path_cost}")
print(small_path)

full_start = (0, 0)
full_goal = (len(full_world[0]) - 1, len(full_world) - 1)
full_path = a_star_search(full_world, full_start, full_goal, COSTS, MOVES, heuristic)
full_path_cost = pretty_print_path(full_world, full_path, full_start, full_goal, COSTS)
print(f"total path cost: {full_path_cost}")
print(full_path)



