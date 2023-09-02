#!/usr/bin/env python
# coding: utf-8

# # Module 6 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from operator import itemgetter
nx.__version__


# **For this assignment only**
# 
# If you want to use NetworkX with your assignment, you can do:
# 
# ```
# conda install networkx
# ```
# or
# 
# ```
# pip install networkx
# ```
# 
# Additionally, this assignment uses a recursive algorithm. You should use `deecopy` at the appropriate places to avoid entanglement issues.

# ## CSP: Map Coloring
# 
# In this programming assignment, you will be using your new understanding of **Constraint Satisfaction Problems** to color maps. As we know from the [Four Color Theorem](http://en.wikipedia.org/wiki/Four_color_theorem) any division of a plane into contiguous regions can be colored such that no two adjacent regions are the same color by using only four colors.
# 
# From the book, we know that we can translate this problem into a CSP where the map is represented as a [planar graph](http://en.wikipedia.org/wiki/Planar_graph) and the goal is to color all the nodes such that no adjacent nodes are colored the same color.
# 
# As with most AI problems, this requires us to figure out how best to represent the problem--and the solution--given the high and low level data structures and types at our disposal. For this problem, we'll settle on a Dict which contains at least two keys: "nodes" which is an *ordered* List of Strings that represents each node or vertex in the planar graph and "edges" which contains a List of Tuples that represent edges between nodes. The Tuples are of ints that represent the index of the node in the "nodes" list.
# 
# Using this system, and adding a "coordinates" key with abstract drawing coordinates of each node for NetworkX, we can represent the counties of Connecticut like so:

# In[2]:


connecticut = { "nodes": ["Fairfield", "Litchfield", "New Haven", "Hartford", "Middlesex", "Tolland", "New London", "Windham"],
                "edges": [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (3,6), (4,6), (5,6), (5,7), (6,7)],
                "coordinates": [( 46, 52), ( 65,142), (104, 77), (123,142), (147, 85), (162,140), (197, 94), (217,146)]}
print(connecticut)


# The coordinates permit us to use NetworkX to draw the graph. We'll add a helper function for this, `draw_map`, which takes our planar_map, a figure size in abstract units, and a List of color assignments in the same order as the nodes in the planar_map.  The underlying drawings are made by matplotlib using NetworkX on top of it. Incidentally, the positions just make the map "work out" on NetworkX/matplotlib.
# 
# The size parameter is actually inches wide by inches tall (8, 10) is an 8x10 sheet of paper. Why doesn't a chart cover up the whole screen then? It's adjusted by dpi. On high resolution monitors, 300 dpi with 8x10 inches might only take up a fraction of that space. Use whatever you want to make the output look good. It doesn't matter for anything else but that.
# 
# A default value for `color_assignments` is provided, `None`, that simply colors all the nodes red. Otherwise, `color_assignments` must be a `List of Tuples` where each `Tuple` is a node name and assigned color. The order of `color_assignments` must be the same as the order of `"nodes"` in the `planar_map`.

# **If you are using NetworkX in your assignment, you can copy this function into module03.py**

# In[3]:


def draw_map(name, planar_map, size, color_assignments=None):
    def as_dictionary(a_list):
        dct = {}
        for i, e in enumerate(a_list):
            dct[i] = e
        return dct
    
    G = nx.Graph()
    
    labels = as_dictionary(planar_map["nodes"])
    pos = as_dictionary(planar_map["coordinates"])
    
    # create a List of Nodes as indices to match the "edges" entry.
    nodes = [n for n in range(0, len(planar_map["nodes"]))]

    if color_assignments:
        colors = [c for n, c in color_assignments]
    else:
        colors = ['red' for c in range(0,len(planar_map[ "nodes"]))]

    G.add_nodes_from(nodes)
    G.add_edges_from(planar_map[ "edges"])

    plt.figure(figsize=size, dpi=100)
    plt.title(name)
    nx.draw_networkx(G, node_color = colors, with_labels = True, labels = labels, pos = pos)
    
#    plt.savefig(name + ".png")


# Using this function, we can draw `connecticut`:

# In[4]:


draw_map("connecticut", connecticut, (5,4), [(n, "red") for n in connecticut["nodes"]])


# This coloring obviously has some problems! You're going to write a program to fix it.

# So what we (and by "we", I mean "you") would like to do is write a program that can figure out how to color a planar map...ie, `connecticut` *and* `europe`, you will do it by implementing a function that uses the algorithms that were discussed in this module.
# 
# ## Which CSP Algorithms?
# 
# You will need to implement **backtracking** and **forward checking**. You can roll your own from Depth First Search or use the pseudocode for `backtracking_search` on p. 192 of AIAMA 4th Edition.
# 
# You will also need to implement one of the following heuristics: **Minimum Remaining Values** or **Degree Heuristic** (tell me which one; it doesn't make sense to implement both) as well as **Least Constraining Value**. Break ties in ascending order (least to most).
# 
# You should get the backtracking and forward checking implemented first. Then add the heuristics. You can use the pseudocode in the book and then stub out the functions that implement the "pick a variable" heuristic and the "pick a value" heuristic. When you get it working, fill in the appropriate heuristic mentioned above.
# 
# Please change the "?" below into "yes" or "no" indicating which elements you were able to implement:

#     backtracking: yes
#     forward checking: yes
#     minimum remaining values: yes
#     degree heuristic: no
#     least contraining value: yes

# Your function should take the following form:
# 
# ```python
# def color_map( planar_map, color_list, trace=False)
# ```
# 
# where `planar_map` has the format described above, colors is a List of Strings denoting the colors to use and `trace` operates as described in the next paragraph. It should return a List of Tuples where each Tuple is of the form `( Node Name, Color)` in the same order as the `node` entry in the planar_map. For example, if we had `["A", "B"]` as nodes and `["Yellow", "Green"]` as colors, your function might return `[("A", "Yellow"), ("B", "Green")]`. If a coloring cannot be found, return `None`.
# 
# Your function also will take an optional argument, `trace`, with a default value of `False`. 
# 
# If `trace` is set to `True` your program will print out *traces* (or debugging) statements that show what it is currently doing (in terms of the algorithms you are supposed to implement). For example, if your program starts to backtrack, the trace should say that it has to backtrack and why.
# 
# As usual, you should implement your function using helper functions, using one Markdown cell for documentation and one Codecell for implementation (one function and assertions).
# 
# -----

# <a id="completeness"></a>
# ## completeness
# 
# This function simply takes in the current assignment of solutions and checks to see if it contains the same amount of answers in the solution **Used by**: [backtracking](#backtracking), [backtracking_basic](#backtracking_basic)
# 
# * **Solution**: list[tuple[str]]: The current solution of the assignment.
# * **Variables**: variables: list[str]: Amount of variables in the world.
# 
# **returns**: Boolean: whether or not the solution has all of the variables in variable

# In[5]:


def completeness(solution: list[tuple[str]], variables: list[str]):
    if len(solution)  == len(variables):
        return True
    else:
        return False


# In[6]:


unit_solution, unit_variables = list(range(0, 9)), list(range(10, 19))
unit_test = completeness(unit_solution, unit_variables)
assert unit_test == True
unit_solution, unit_variables = list(range(0, 119)), list(range(10, 19))
unit_test = completeness(unit_solution, unit_variables)
assert unit_test == False
unit_solution, unit_variables = list(range(1, 1000)), list(range(1000, 1999))
unit_test = completeness(unit_solution, unit_variables)
assert unit_test == True


# <a id="basic_select_variable"></a>
# ## basic_select_variable
# 
# This function is a simply implementation of selecting the next variable to determine its domain selection. Essentially it is used to find the remaining variables that do not have an assignment and return the first one **Used by**: [backtracking_basic](#backtracking_basic)
# 
# * **Solution**: list[tuple[str]]: The current solution of the assignment.
# * **Variables**: variables: list[str]: Amount of variables in the world.
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: str: returns the node that should be looked at next.

# In[7]:


def basic_select_variable(solution: list[tuple[str]], variables: list[str], trace:bool=False):
    remaining_var  = []
    for node in variables:
        if node not in solution:
            remaining_var.append(node)
    location = remaining_var[0]
    if trace:
        print("The node selected is: ", remaining_var[0])
    return location


# In[8]:


unit_solution = {"Fairfield": "red"}
unit_variables = connecticut["nodes"]
unit_test = basic_select_variable(unit_solution, unit_variables)
assert unit_test == "Litchfield"
unit_solution = {"Fairfield": "red", "Litchfield": "blue"}
unit_variables = connecticut["nodes"]
unit_test = basic_select_variable(unit_solution, unit_variables)
assert unit_test == "New Haven"
unit_solution = {"Fairfield": "red", "Litchfield": "blue", "New Haven": "green", "Hartford": "purple", "Middlesex": "orange"}
unit_variables = connecticut["nodes"]
unit_test = basic_select_variable(unit_solution, unit_variables, True)
assert unit_test == "Tolland"


# <a id="minimum_remaining_values"></a>
# ## minimum_remaining_values
# 
# This function is also used to determine which variable is looked at next, except in this case, it is comparing how many values are left per variable, then selecting the one that has the least remaining values available to it in the domain **Used by**: [backtracking](#backtracking).
# 
# * **assignments**: list[tuple[str]]: The current solution of the assignment.
# * **variables**: list[str]: Amount of variables in the world.
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: str: returns the node that should be looked at next.

# In[9]:


def minimum_remaining_values(assignments: list[tuple[str]], variables: list[str], domains: dict[str, list[str]], trace=False):
    remaining_var, var_available_domain  = [], []
    for node in variables:
        if node not in assignments:
            remaining_var.append(node)
    for var in remaining_var:
        var_available_domain.append((var, len(domains[var])))
    variables_left = sorted(var_available_domain, key=itemgetter(1))
    if trace:
        print("MRV in use, selected", variables_left[0][0], "with the amount of values", variables_left[0][1])
    return variables_left[0][0]


# In[10]:


unit_solution = {"Fairfield": "red"}
unit_variables = connecticut["nodes"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = ["red", "blue", "green", "purple", "orange"]
unit_test = minimum_remaining_values(unit_solution, unit_variables, unit_domains)
assert unit_test == "Litchfield"
unit_solution = {"Fairfield": "red", "Litchfield": "blue"}
unit_variables = connecticut["nodes"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = ["red", "blue", "green", "purple", "orange"]
unit_test = minimum_remaining_values(unit_solution, unit_variables, unit_domains)
assert unit_test == "New Haven"
unit_solution = {"Fairfield": "red", "Litchfield": "blue", "New Haven": "green", "Hartford": "purple", "Middlesex": "orange"}
unit_variables = connecticut["nodes"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = ["red", "blue", "green", "purple", "orange"]
unit_test = minimum_remaining_values(unit_solution, unit_variables, unit_domains, True)
assert unit_test == "Tolland"


# <a id="create_edge_list"></a>
# ## create_edge_list
# 
# This function takes in the edges and the variables that are used to determine what the edges are connecting, then returns a dict that contains constraints on specific variables that will be used for determine who cannot share values **Used by**: [color_map](#color_map)
# 
# * **nodes**: list[str]: Amount of variables in the world.
# * **edges**: Tuple[int, int]: this a set of tuples that show the edges between specific locations in the world
# 
# **returns**: dict[str, list[str]]: returns a dict that determines the constraints that cannot be shared on different nodes

# In[11]:


def create_edge_list(nodes: list[str], edges: tuple[int, int]):
    edge_dict = {}
    for node in nodes:
        nodes_in_edge = []
        for x, y in edges:
            if nodes[x] != node and nodes[y] == node:
                nodes_in_edge.append(nodes[x])
            if nodes[y] != node and nodes[x] == node:
                nodes_in_edge.append(nodes[y])
        edge_dict[node] = nodes_in_edge
    return edge_dict


# In[12]:


unit_edge_list = create_edge_list(connecticut["nodes"], connecticut["edges"])
assert unit_edge_list == {'Fairfield': ['Litchfield', 'New Haven'], 'Litchfield': ['Fairfield', 'New Haven', 'Hartford'], 'New Haven': ['Fairfield', 'Litchfield', 'Hartford', 'Middlesex'], 'Hartford': ['Litchfield', 'New Haven', 'Middlesex', 'Tolland', 'New London'], 'Middlesex': ['New Haven', 'Hartford', 'New London'], 'Tolland': ['Hartford', 'New London', 'Windham'], 'New London': ['Hartford', 'Middlesex', 'Tolland', 'Windham'], 'Windham': ['Tolland', 'New London']}
unit_edge_list = create_edge_list(connecticut["nodes"][:5], connecticut["edges"][:5])
assert unit_edge_list == {'Fairfield': ['Litchfield', 'New Haven'], 'Litchfield': ['Fairfield', 'New Haven', 'Hartford'], 'New Haven': ['Fairfield', 'Litchfield', 'Hartford'], 'Hartford': ['Litchfield', 'New Haven'], 'Middlesex': []}
unit_edge_list = create_edge_list(["test", "testing", "tested"], [(0, 1), (0, 2), (1, 2)])
assert unit_edge_list == {'test': ['testing', 'tested'], 'testing': ['test', 'tested'], 'tested': ['test', 'testing']}


# <a id="least_constraining_values"></a>
# ## least_constraining_values
# 
# This is a heuristic to determine what values are the least constrained in the given selections by looking at each value with given constraints and determine which of them have the least constraining constraints applied to them **Used by**: [backtracking](#backtracking).
# 
# * **var**: str: the specific variable that will be looked at to assign a value
# * **assignments**: list[tuple[str]]: The current solution of the assignment.
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **constraints**: dict[str, list[str]]: the limitations on shared domains for specific nodes
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: list[str]: returns a list of values in order of least constraining to the backtracking search to determine which value should be sent next.

# In[13]:


def least_constraining_values(var: str, assignments: list[tuple[str]], domains: dict[str, list[str]], constraints: dict[str, list[str]], trace:bool=False):
    next_to, values, common_domains = constraints[var], [], []
    for value in domains[var]:
        count = 0
        for neighbor in next_to:
            if value in domains[neighbor]:
                count += 1
        values.append((value, count))
    sorted_common_value = sorted(values, key=itemgetter(1), reverse=False)
    for x, y in sorted_common_value:
        common_domains.append(x)
    if trace:
            print("Using LCV, selected", common_domains, "For the variable", var)
    return common_domains


# In[14]:


unit_var = "Fairfield"
unit_solution = {"Fairfield": "red"}
unit_domain1 = dict([(x, ["red", "blue", "green", "yellow"]) for x in connecticut["nodes"]])
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_test_list = []
for unit_test in least_constraining_values(unit_var, unit_solution, unit_domain1, unit_constraints, trace=True):
    unit_test_list.append(unit_test)
assert unit_test_list == ['red', 'blue', 'green', 'yellow']
unit_var = "New Haven"
unit_solution = {"Fairfield": "red", "Litchfield": "blue"}
unit_domain1 = dict([(x, ["test", "tested", "this is unit test"]) for x in connecticut["nodes"]])
unit_test_list = []
for unit_test in least_constraining_values(unit_var, unit_solution, unit_domain1, unit_constraints):
    unit_test_list.append(unit_test)
assert unit_test_list == ['test', 'tested', 'this is unit test']
unit_var = "Fairfield"
unit_solution = {"Fairfield": "red", "Litchfield": "blue", "New Haven": "green", "Hartford": "purple", "Middlesex": "orange"}
unit_domain1 = dict([(x, ["test"]) for x in connecticut["nodes"]])
unit_test_list = []
for unit_test in least_constraining_values(unit_var, unit_solution, unit_domain1, unit_constraints):
    unit_test_list.append(unit_test)
assert unit_test_list == ['test']


# <a id="basic_domain_selection"></a>
# ## basic_domain_selection
# 
# This is a basic function that does not use any heuristics or anything, just assigns the domain to a variable without determining any best use cases **Used by**: [backtracking_basic](#backtracking_basic).
# 
# * **var**: str: the specific variable that will be looked at to assign a value
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: list[str]: returns a list of values in order of least constraining to the backtracking search to determine which value should be sent next.

# In[15]:


def basic_domain_selection(var: str, domains: dict[str, list[str]], trace:bool=False):
    options = []
    for domain_selection in domains[var]:
        options.append(domain_selection)
    if trace == True:
            print("Selected Value", options, "for", var)
    return options
    


# In[16]:


unit_var = "Fairfield"
unit_domain1 = dict([(x, ["red", "blue", "green", "yellow"]) for x in connecticut["nodes"]])
unit_test_list = []
unit_test_list = basic_domain_selection(unit_var, unit_domain1, trace=True)
assert unit_test_list == ['red', 'blue', 'green', 'yellow']
unit_var = "New Haven"
unit_domain1 = dict([(x, ["test", "tested", "this is unit test"]) for x in connecticut["nodes"]])
unit_test_list = []
for unit_test in basic_domain_selection(unit_var, unit_domain1):
    unit_test_list.append(unit_test)
assert unit_test_list == ['test', 'tested', 'this is unit test']
unit_var = "test"
unit_domain1 = dict([(x, ["test", "tested", "this is unit test"]) for x in ["testing", "tested", "test"]])
unit_test_list = []
for unit_test in basic_domain_selection(unit_var, unit_domain1):
    unit_test_list.append(unit_test)
assert unit_test_list == ['test', 'tested', 'this is unit test']


# <a id="check_consistent"></a>
# ## check_consistent
# 
# This function determines whether or not the given variable and value is consistent with given constraints that are assigned. **Used by**: [backtracking](#backtracking), [backtracking_basic](#backtracking_basic).
# 
# * **var**: str: the specific variable that will be looked at to assign a value
# * **value**: str: the assigned domain to the variable node
# * **assignments**: list[tuple[str]]: The current solution of the assignment.
# * **constraints**: dict[str, list[str]]: the limitations on shared domains for specific nodes
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: documentation of the returned value and type.

# In[17]:


def check_consistent(var: str, value: str, assignments: list[tuple[str]], constraints: dict[str, list[str]], trace:bool=False):
    if assignments is None:
        return False
    for limit in constraints:
        if limit in assignments:
            if value is assignments[limit]:
                if trace:
                    print(var, "assignment ", value, " has conflictions with ", limit, assignments[limit])
                return False
    return True


# In[18]:


unit_variable = basic_select_variable("Fairfield", connecticut["nodes"])
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_value = "red"
unit_assignments = []
unit_test = check_consistent(unit_variable, unit_value, unit_assignments, unit_constraints)
assert unit_test == True
unit_variable = basic_select_variable("New Haven", connecticut["nodes"])
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_value = "blue"
unit_assignments = {"Fairfield" : "blue"}
unit_test = check_consistent(unit_variable, unit_value, unit_assignments, unit_constraints)
assert unit_test == False
unit_variable = basic_select_variable("New Haven", connecticut["nodes"])
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_value = "blue"
unit_assignments = {"Fairfield" : "red"}
unit_test = check_consistent(unit_variable, unit_value, unit_assignments, unit_constraints)
assert unit_test == True


# <a id="forward_check"></a>
# ## forward_check
# 
# This function takes in the variables, domains, constraints, all parts of the csp, and changes the forward domains of what will be checked in the future to reduce what is already being used by the current domain, that way should issues appear in the future they can be stopped quicker. If a future variable already has no values left in the domain, then it will return false, saving time by not even looking into the variable with no values left to select **Used by**: [backtracking](#backtracking), [backtracking_basic](#backtracking_basic).
# 
# * **variables**: list[str]: Amount of variables in the world.
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **constraints**: dict[str, list[str]]: the limitations on shared domains for specific nodes
# * **var**: str: the specific variable that will be looked at to assign a value
# * **value**: str: the assigned domain to the variable node
# * * **assignments**: list[tuple[str]]: The current solution of the assignment.
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: bool: returns if a domain still has values in it for true, and false if no other domains available

# In[19]:


def forward_check(variables, domains, constraints, var, value, assignments, trace=False):
    neighbors = [next for next in constraints[var]]
    for neighbor in neighbors:
        if value in domains[neighbor]:
            if trace: print(value, "not in domain for", neighbor, "removing")
            available_domain = []
            for available in domains[neighbor]:
                if available != value:
                    available_domain.append(available)
            domains[neighbor] = available_domain
        if len(domains[neighbor]) == 0:
            return False
        return True


# In[20]:


unit_variables, unit_domains, unit_var, unit_value, unit_assignments = connecticut["nodes"], {}, "Fairfield", "red", {}
for node in unit_variables:
    unit_domains[node] = ["red", "blue", "green", "purple"]
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_test = forward_check(unit_variables, unit_domains, unit_constraints, unit_var, unit_value, unit_assignments)
assert unit_test == True
unit_variables, unit_domains, unit_var, unit_value, unit_assignments = connecticut["nodes"], {}, "New Haven", "red", {"Fairfield" : "red"}
for node in unit_variables:
    unit_domains[node] = []
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_test = forward_check(unit_variables, unit_domains, unit_constraints, unit_var, unit_value, unit_assignments)
assert unit_test == False
unit_variables, unit_domains, unit_var, unit_value, unit_assignments = connecticut["nodes"], {}, "New Haven", "red", {}
for node in unit_variables:
    unit_domains[node] = ["red", "blue", "green", "purple"]
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_test = forward_check(unit_variables, unit_domains, unit_constraints, unit_var, unit_value, unit_assignments, trace=True)
assert unit_test == True


# <a id="backtracking_search"></a>
# ## backtracking_search
# 
# Backtracking is a recursive algorithm, and this function simply calls backtracking to be used **Uses**: [backtracking](#backtracking) OR [backtracking_basic](#backtracking_basic) **Used by**: [color_map](#color_map).
# 
# * **variables**: list[str]: Amount of variables in the world.
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **constraints**: dict[str, list[str]]: the limitations on shared domains for specific nodes
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: bool, list[tuple[str, str]]: this function returns a bool to state whether or not the solution is valid or not, then the actual solution which contains a node and the value that was assigned to it.

# In[21]:


def backtracking_search(variables: list[str], domains: dict[str, list[str]], constraints: dict[str, list[str]], trace=False):
    solution = {}
    return backtracking(solution, variables, domains, constraints, trace=trace)


# Since this just calls backtracking, I did not think unit tests were worth doing, this is just a helper function to call another

# <a id="backtracking_basic"></a>
# ## backtracking_basic
# 
# This function is a basic implementation of backtracking where no heuristics are used in determining variables and values, simply grabbing the first one then uses forward checking and backtracking to consistently determine consistent and complete solutions to coloring the world. **Uses**: [basic_domain_selection](#basic_domain_selection), [basic_select_variable](#basic_select_variable), [check_consistent](#check_consistent), [completeness](#completeness), [forward_check](#forward_check). **Used by**: [backtracking_search](#backtracking_search).
# 
# * **assignments**: list[tuple[str]]: The current solution of the assignment.
# * **variables**: list[str]: Amount of variables in the world.
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **constraints**: dict[str, list[str]]: the limitations on shared domains for specific nodes
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: bool, list[tuple[str, str]]: this function returns a bool to state whether or not the solution is valid or not, then the actual solution which contains a node and the value that was assigned to it.
# 

# In[22]:


def backtracking_basic(assignments: list[tuple[str]], variables: list[str], domains: dict[str, list[str]], constraints: dict[str, list[str]], trace:bool=False):
    if trace: print("Solution = ", assignments)
    if completeness(assignments, variables):
        return True, assignments
    var = basic_select_variable(assignments, variables, trace=trace)
    for value in basic_domain_selection(var, domains, trace=trace):
        if check_consistent(var, value, assignments, constraints[var], trace=trace):
            assignments[var] = value
            m_variables, m_domains, m_constraints = deepcopy(variables), deepcopy(domains), deepcopy(constraints)
            if forward_check(variables, domains, constraints, var, value, assignments, trace):
                valid, assignments = backtracking_basic(deepcopy(assignments), variables, domains, constraints, trace)
                if completeness(assignments, variables):
                    return True, assignments
                if valid:
                    return True, assignments
            variables, domains, constraints = m_variables, m_domains, m_constraints
    return False, assignments


# In[23]:


unit_assignments = {}
unit_variables = connecticut["nodes"]
unit_colors = ["purple", "red", "green", "yellow"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = unit_colors
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_valid, unit_solution = backtracking_basic(unit_assignments, unit_variables, unit_domains, unit_constraints)
assert unit_solution == {'Fairfield': 'purple', 'Litchfield': 'red', 'New Haven': 'green', 'Hartford': 'purple', 'Middlesex': 'red', 'Tolland': 'red', 'New London': 'green', 'Windham': 'purple'}
unit_assignments = {}
unit_variables = connecticut["nodes"]
unit_colors = []
unit_colors = ["dirty bubble", "mermaid man", "barnacle boy", "purple"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = unit_colors
unit_valid, unit_solution = backtracking_basic(unit_assignments, unit_variables, unit_domains, unit_constraints)
assert unit_solution == {'Fairfield': 'dirty bubble', 'Litchfield': 'mermaid man', 'New Haven': 'barnacle boy', 'Hartford': 'dirty bubble', 'Middlesex': 'mermaid man', 'Tolland': 'mermaid man', 'New London': 'barnacle boy', 'Windham': 'dirty bubble'}
unit_assignments = {}
unit_variables = connecticut["nodes"]
unit_colors = []
unit_colors = ["purple", "red", "green"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = unit_colors
unit_valid, unit_solution = backtracking_basic(unit_assignments, unit_variables, unit_domains, unit_constraints)
assert unit_solution == {'Fairfield': 'purple', 'Litchfield': 'red', 'New Haven': 'green', 'Hartford': 'purple', 'Middlesex': 'red', 'Tolland': 'red', 'New London': 'green', 'Windham': 'purple'}


# <a id="backtracking"></a>
# ## backtracking
# 
# This is an implementation of backtracking where it uses the mrv and lcv functions to determine values and variables to be used for forward checking and backtracking as an attempt to improve efficency in the algorithm for finding consistent and complete assignments. **Uses**: [least_constraining_value](#least_constraining_value), [minimum_remaining_value](#minimum_remaining_value), [check_consistent](#check_consistent), [completeness](#completeness), [forward_check](#forward_check). **Used by**: [backtracking_search](#backtracking_search).
# 
# * **assignments**: list[tuple[str]]: The current solution of the assignment.
# * **variables**: list[str]: Amount of variables in the world.
# * **domains**: dict[str, list[str]]: the amount of values possible to assign to a variable
# * **constraints**: dict[str, list[str]]: the limitations on shared domains for specific nodes
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: bool, list[tuple[str, str]]: this function returns a bool to state whether or not the solution is valid or not, then the actual solution which contains a node and the value that was assigned to it.

# In[24]:


def backtracking(assignments: list[tuple[str]], variables: list[str], domains: dict[str, list[str]], constraints: dict[str, list[str]], trace:bool=False):
    if trace: print("Solution = ", assignments)
    if completeness(assignments, variables):
        return True, assignments
    var = minimum_remaining_values(assignments, variables, domains, trace)
    for value in least_constraining_values(var, assignments, domains, constraints, trace):
        if check_consistent(var, value, assignments, constraints[var], trace):
            assignments[var] = value
            m_variables, m_domains, m_constraints = deepcopy(variables), deepcopy(domains), deepcopy(constraints)
            if forward_check(variables, domains, constraints, var, value, assignments, trace):
                valid, assignments = backtracking(deepcopy(assignments), variables, domains, constraints, trace)
                if completeness(assignments, variables):
                    return True, assignments
                if valid:
                    return True, assignments
            del assignments[var]
            variables, domains, constraints = m_variables, m_domains, m_constraints
    return False, assignments


# In[25]:


unit_assignments = {}
unit_variables = connecticut["nodes"]
unit_colors = ["purple", "red", "green", "yellow"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = unit_colors
unit_constraints = create_edge_list(connecticut["nodes"], connecticut["edges"])
unit_valid, unit_solution = backtracking(unit_assignments, unit_variables, unit_domains, unit_constraints)
assert unit_solution == {'Fairfield': 'purple', 'Litchfield': 'red', 'New Haven': 'green', 'Hartford': 'purple', 'Middlesex': 'red', 'Tolland': 'red', 'New London': 'green', 'Windham': 'purple'}
unit_assignments = {}
unit_variables = connecticut["nodes"]
unit_colors = []
unit_colors = ["dirty bubble", "mermaid man", "barnacle boy", "purple"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = unit_colors
unit_valid, unit_solution = backtracking(unit_assignments, unit_variables, unit_domains, unit_constraints)
assert unit_solution == {'Fairfield': 'dirty bubble', 'Litchfield': 'mermaid man', 'New Haven': 'barnacle boy', 'Hartford': 'dirty bubble', 'Middlesex': 'mermaid man', 'Tolland': 'mermaid man', 'New London': 'barnacle boy', 'Windham': 'dirty bubble'}
unit_assignments = {}
unit_variables = connecticut["nodes"]
unit_colors = []
unit_colors = ["purple", "red", "green"]
unit_domains = {}
for node in unit_variables:
        unit_domains[node] = unit_colors
unit_valid, unit_solution = backtracking(unit_assignments, unit_variables, unit_domains, unit_constraints)
assert unit_solution == {'Fairfield': 'purple', 'Litchfield': 'red', 'New Haven': 'green', 'Hartford': 'purple', 'Middlesex': 'red', 'Tolland': 'red', 'New London': 'green', 'Windham': 'purple'}


# <a id="color_map"></a>
# ## color_map
# 
# This function takes in a dict of nodes that have edges and coordinates to be mapped by draw_map. The goal is for this to really start the CSP process by calling backtracking_search after changing the information into an easy to use format for backtracking to successfully run correctly. Once that is finished it will return the node iwth an assigned value to it.  **Uses**: [backtracking_search](#backtracking_search)
# 
# * **planar_map**: dict[str, list]: a dict 
# * **color_list**: list[str]: documentation of parameter and type.
# * **trace**: boolean: contains whether or not we should print information to see what the algorithm is doing
# 
# **returns**: list[tuple[str, str]]: Returns a list of tuples that contain the node and the color it was assigned

# In[26]:


def color_map(planar_map: dict[str, list], color_list: list[str], trace:bool=False):
    variables, domains, solution = planar_map["nodes"], {}, []
    for node in variables:
        domains[node] = color_list
    constraints = create_edge_list(planar_map["nodes"], planar_map["edges"])
    valid, answer = backtracking_search(variables, domains, constraints, trace=trace)
    for assign in answer:
        solution.append((assign, answer[assign]))
    if not valid:
        return None
    else:
        return solution


# Currently, it just colors everything red. When you are done, if it cannot find a coloring, it should return `None`.

# ## Problem 1. Color Connecticut Using Your Solution

# In[27]:


connecticut_colors = color_map(connecticut, ["red", "blue", "green", "yellow"], trace=True)


# Using the "edges" list from the connecticut map, we can test to see if each pair of adjacent nodes is indeed colored differently:

# In[28]:


edges = connecticut["edges"]
nodes = connecticut[ "nodes"]
colors = connecticut_colors
COLOR = 1

for start, end in edges:
    try:
        assert colors[start][COLOR] != colors[end][COLOR]
    except AssertionError:
        print(f"{nodes[start]} and {nodes[end]} are adjacent but have the same color.")


# In[29]:


draw_map("connecticut", connecticut, (5,4), connecticut_colors)


# In[30]:


connecticut_colors = color_map( connecticut, ["red", "blue", "green"], trace=True)
if connecticut_colors:
    draw_map("connecticut", connecticut, (5,4), connecticut_colors)


# ## Problem 2. Color Europe Using Your solution

# In[31]:


europe = {
    "nodes":  ["Iceland", "Ireland", "United Kingdom", "Portugal", "Spain",
                 "France", "Belgium", "Netherlands", "Luxembourg", "Germany",
                 "Denmark", "Norway", "Sweden", "Finland", "Estonia",
                 "Latvia", "Lithuania", "Poland", "Czech Republic", "Austria",
                 "Liechtenstein", "Switzerland", "Italy", "Malta", "Greece",
                 "Albania", "Macedonia", "Kosovo", "Montenegro", "Bosnia Herzegovina",
                 "Serbia", "Croatia", "Slovenia", "Hungary", "Slovakia",
                 "Belarus", "Ukraine", "Moldova", "Romania", "Bulgaria",
                 "Cyprus", "Turkey", "Georgia", "Armenia", "Azerbaijan",
                 "Russia" ], 
    "edges": [(0,1), (0,2), (1,2), (2,5), (2,6), (2,7), (2,11), (3,4),
                 (4,5), (4,22), (5,6), (5,8), (5,9), (5,21), (5,22),(6,7),
                 (6,8), (6,9), (7,9), (8,9), (9,10), (9,12), (9,17), (9,18),
                 (9,19), (9,21), (10,11), (10,12), (10,17), (11,12), (11,13), (11,45), 
                 (12,13), (12,14), (12,15), (12,17), (13,14), (13,45), (14,15),
                 (14,45), (15,16), (15,35), (15,45), (16,17), (16,35), (17,18),
                 (17,34), (17,35), (17,36), (18,19), (18,34), (19,20), (19,21), 
                 (19,22), (19,32), (19,33), (19,34), (20,21), (21,22), (22,23),
                 (22,24), (22,25), (22,28), (22,29), (22,31), (22,32), (24,25),
                 (24,26), (24,39), (24,40), (24,41), (25,26), (25,27), (25,28),
                 (26,27), (26,30), (26,39), (27,28), (27,30), (28,29), (28,30),
                 (29,30), (29,31), (30,31), (30,33), (30,38), (30,39), (31,32),
                 (31,33), (32,33), (33,34), (33,36), (33,38), (34,36), (35,36),
                 (35,45), (36,37), (36,38), (36,45), (37,38), (38,39), (39,41),
                 (40,41), (41,42), (41,43), (41,44), (42,43), (42,44), (42,45),
                 (43,44), (44,45)],
    "coordinates": [( 18,147), ( 48, 83), ( 64, 90), ( 47, 28), ( 63, 34),
                   ( 78, 55), ( 82, 74), ( 84, 80), ( 82, 69), (100, 78),
                   ( 94, 97), (110,162), (116,144), (143,149), (140,111),
                   (137,102), (136, 95), (122, 78), (110, 67), (112, 60),
                   ( 98, 59), ( 93, 55), (102, 35), (108, 14), (130, 22),
                   (125, 32), (128, 37), (127, 40), (122, 42), (118, 47),
                   (127, 48), (116, 53), (111, 54), (122, 57), (124, 65),
                   (146, 87), (158, 65), (148, 57), (138, 54), (137, 41),
                   (160, 13), (168, 29), (189, 39), (194, 32), (202, 33),
                   (191,118)]}
print(europe)


# In[32]:


europe_colors = color_map(europe, ["red", "blue", "green", "yellow"], trace=True)


# Here we're testing to see if the adjacent nodes are colored differently:

# In[33]:


edges = europe["edges"]
nodes = europe[ "nodes"]
colors = europe_colors
COLOR = 1

for start, end in edges:
    try:
        assert colors[start][COLOR] != colors[end][COLOR]
    except AssertionError:
        print(f"{nodes[start]} and {nodes[end]} are adjacent but have the same color.")


# In[34]:


draw_map("europe", europe, (10,8), europe_colors)


# In[35]:


europe_colors = color_map(europe, ["red", "blue", "green"], trace=True)
if europe_colors:
     draw_map("europe", europe, (10,8), europe_colors)


# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# The below is still good information but by the time I turned in the assignment I did get MRV and LCV working to a good state reporting the correct answer. There was issues with how I set up my forward checking where it would remove the backedge of a variable, so everytime it ran it ended up just forgetting about any edges that Came before it. For example, New Haven would end up ignoring Fairfield. Once that was resolved it worked well. Oddly, though, having no heuristics at all did not care that was happening and it would work as expected anyways. I guess because its mostly selecting in a consistent order.
# 
# My logging may have been a bit long, so I apologize for that. 

# This was more of for my benefit, but here was the results of running color_map using a basic selections for the domain and variable that just chose the first one. I was able to get this running running fairly quickly, but struggled with getting LCV and MRV working at one point in time. I essentially created a second backtracking just called backtracking_basic that I would change the backtracking to use that, but did not want to many duplicates so in order to use it I needed to change backtracking_search to call it instead of backtracking. 

# In[482]:


connecticut_colors = color_map( connecticut, ["red", "blue", "green"], trace=False)
edges = connecticut["edges"]
nodes = connecticut[ "nodes"]
colors = connecticut_colors
COLOR = 1

for start, end in edges:
    try:
        assert colors[start][COLOR] != colors[end][COLOR]
    except AssertionError:
        print(f"{nodes[start]} and {nodes[end]} are adjacent but have the same color.")
if connecticut_colors:
    draw_map("connecticut", connecticut, (5,4), connecticut_colors)

europe_colors = color_map(europe, ["red", "blue", "green", "yellow"], trace=False)
edges = europe["edges"]
nodes = europe[ "nodes"]
colors = europe_colors
COLOR = 1

for start, end in edges:
    try:
        assert colors[start][COLOR] != colors[end][COLOR]
    except AssertionError:
        print(f"{nodes[start]} and {nodes[end]} are adjacent but have the same color.")
draw_map("europe", europe, (10,8), europe_colors)


# In[ ]:




