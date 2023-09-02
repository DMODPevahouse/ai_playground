from unification import parse, unification
from copy import deepcopy
import pprint
self_check_state = [
    "(plane 1973)",
    "(plane 2749)",
    "(plane 97)",
    "(plane 1211)",
    "(airport SFO)",
    "(airport JFK)",
    "(airport ORD)",
    "(at 1973 SFO)",
    "(at 2749 JFK)",
    "(at 97 ORD)",
    "(at 1211 SFO)",
    "(fueled 1973)",
    "(unfueled 2749)",
    "(unfueled 97)",
    "(fueled 1211)"
]
self_check_actions = {
    "fly": {
        "action": "(fly ?plane ?from ?to)",
        "conditions": [
            "(plane ?plane)",
            "(airport ?to)",
            "(airport ?from)",
            "(at ?plane ?from)"
            "(fueled ?plane)"
        ],
        "add": [
            "(at ?plane ?to)",
            "(unfueled ?plane)"
        ],
        "delete": [
            "(at ?plane ?from)",
            "(fueled ?plane)"
        ]
    },
    "fuel": {
        "action": "(fueled ?plane)",
        "conditions": [
            "(plane ?plane)",
            "(unfueled ?plane)"
        ],
        "add": [
            "(fueled ?plane)"
        ],
        "delete": [
            "(unfueled ?plane)"
        ]
    }
}


# <a id="is_goal"></a>
# ## is_goal
#
# This algorithm, backtracking, does have a recursive element to it, so this simple function is really just used for testing to see if the goal state is reached **Used by**: [action_checking](#action_checking), [action_backtracking](#action_backtracking)
#
# * **current_state**: list[list]: The current state of the search.
# * **goal**: list[list]: The state that we want to have a plan to achieve.
#
# **returns**: bool: of if the goal is met or not by the actions that have changed it

# In[6]:


def is_goal(current_state: list[list], goal: list[list]):
    return all([True if line in goal else False for line in current_state])


# In[7]:


unit_goal = [
    "(plane 1973)",
    "(plane 2749)",
    "(plane 97)",
    "(plane 1211)",
    "(airport SFO)",
    "(airport JFK)",
    "(airport ORD)",
    "(at 1973 SFO)",
    "(at 2749 JFK)",
    "(at 97 ORD)",
    "(at 1211 SFO)",
    "(fueled 1973)",
    "(unfueled 2749)",
    "(unfueled 97)",
    "(fueled 1211)"
]
unit_self_check_state = deepcopy(self_check_state)
for unit_line in unit_goal:
    unit_line = parse(unit_line)
for unit_line in unit_self_check_state:
    unit_line = parse(unit_line)
unit_test = is_goal(unit_self_check_state, unit_goal)
assert unit_test == True
unit_goal = [
    "(plane 1973)",
    "(plane 2749)",
    "(plane 97)",
    "(plane 1211)",
    "(airport SFO)"
]
unit_test = is_goal(self_check_state[0:4], unit_goal)
assert unit_test == True
unit_goal = [
    "(plane 1973)",
    "(plane 2749)",
    "(plane 97)",
    "(plane 1211)",
    "(airport SFO)"
]
unit_test = is_goal(self_check_state, unit_goal)
assert unit_test == False


# <a id="parse_actions"></a>
# ## parse_actions
#
# This is a simple function that will take the actions given the syntax that is used in previous unification assignments and parse each item into a list format that is more usable for the given unification algorithm **Used by**: [forward_planning](#forward_planning)
#
# * **actions**: dict: The original actions with the format of a single string in parenthesis, and others can be in paranthesis in there as well.
#
#
# **returns**: dict: based on the original but now in list format instead of string format

# In[8]:


def parse_actions(actions):
    for key in actions:
        actions[key]["action"] = parse(actions[key]["action"])
        for i, condition in enumerate(actions[key]["conditions"]):
            actions[key]["conditions"][i] = parse(condition)
        for i, add in enumerate(actions[key]["add"]):
            actions[key]["add"][i] = parse(add)
        for i, delete in enumerate(actions[key]["delete"]):
            actions[key]["delete"][i] = parse(delete)
    return actions


# In[9]:


unit_test = parse_actions(deepcopy(self_check_actions))
assert unit_test['fly']['action'] == ['fly', '?plane', '?from', '?to']
assert unit_test['fly']['conditions'] == [['plane', '?plane'],
                                          ['airport', '?to'],
                                          ['airport', '?from'],
                                          ['at', '?plane', '?from']]
assert unit_test['fly']['add'] == [['at', '?plane', '?to'], ['unfueled', '?plane']]
assert unit_test['fly']['delete'] == [['at', '?plane', '?from'], ['fueled', '?plane']]
assert unit_test['fuel']['action'] == ['fueled', '?plane']
assert unit_test['fuel']['conditions'] == [['plane', '?plane'], ['unfueled', '?plane']]
assert unit_test['fuel']['add'] == [['fueled', '?plane']]
assert unit_test['fuel']['delete'] == [['unfueled', '?plane']]


# <a id="variable_substitution"></a>
# ## variable_substitution
#
#  This functions main purpose is to take a precondition with a given expression, that is a dict, and uses that variable and value in that expression to populate the precondition with that value in order to easily test and use it in the algorithm **Used by**: [forward_checking](#forward_checking), [create_filled_action](#create_filled_action)
#
# * **to_be_substituted**: list: this is the condition that needs to be met .
# * **expression**: dict: the variables and values in a dict format that will be converted to set up the substitution.
#
# **returns**: list: returns the same list except the previous variables are now the assigned values

# In[10]:


def variable_substitution(to_be_substituted: list, expression: dict):
    variable, value = [], []
    if len(expression) == 0 or len(to_be_substituted) == 0:
        return to_be_substituted
    for key in expression:
        value.append(expression[key])
        variable.append(key)
    for j, specific in enumerate(expression):
        for i, element in enumerate(to_be_substituted):
            if isinstance(element, list):
                to_be_substituted[i] = variable_substitution(element, expression)
            elif variable[j] in element:
                to_be_substituted[i] = value[j]
    return to_be_substituted


# In[11]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
unit_test = variable_substitution(unit_test_action["fly"]["action"], {"?plane": "54", "?from": "STL", "?to": "LA"})
assert unit_test == ['fly', '54', 'STL', 'LA']
unit_test = variable_substitution(unit_test_action["fuel"]["action"], {"?plane": "11223"})
assert unit_test == ["fueled", "11223"]
unit_test = variable_substitution(unit_test_action["fly"]["conditions"],
                                  {"?plane": "94", "?to": "STL", "?from": "MD", "?airport": "LAM"})
assert unit_test == [['plane', '94'], ['airport', 'STL'], ['airport', 'MD'], ['at', '94', 'MD']]


# <a id="create_filled_action"></a>
# ## create_filled_action
#
# This function only purpose is just to more cleaning take an action and use the function variable_substituation to make sure and fill the action with the appropriate variables for the current state that is being checked **Uses**: [variable_substitution](#variable_substitution) **Used by**: [condition_backtracking](#condition_backtracking)
#
# * **expression**: dict: the variables and values in a dict format that will be converted to set up the substitution.
# * **action**: dict: This is a specific action that will be in format of variables rather then values.
# * (add more as necessary)
#
# **returns**: dict: this function returns the same action except now the variables are values

# In[12]:


def create_filled_action(expression: dict, action: dict):
    filled_action = deepcopy(action)
    filled_action["action"] = variable_substitution(filled_action["action"], expression)
    filled_action["conditions"] = variable_substitution(filled_action["conditions"], expression)
    filled_action["add"] = variable_substitution(filled_action["add"], expression)
    filled_action["delete"] = variable_substitution(filled_action["delete"], expression)
    return filled_action


# In[13]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
unit_test = create_filled_action({"?plane": "94", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                 deepcopy(unit_test_action["fly"]))
assert unit_test == {'action': ['fly', '94', 'MD', 'STL'],
                     'conditions': [['plane', '94'],
                                    ['airport', 'STL'],
                                    ['airport', 'MD'],
                                    ['at', '94', 'MD']],
                     'add': [['at', '94', 'STL'], ['unfueled', '94']],
                     'delete': [['at', '94', 'MD'], ['fueled', '94']]}
unit_test = create_filled_action({"?plane": "54", "?from": "STL", "?to": "LA"}, deepcopy(unit_test_action["fly"]))
assert unit_test == {'action': ['fly', '54', 'STL', 'LA'],
                     'conditions': [['plane', '54'],
                                    ['airport', 'LA'],
                                    ['airport', 'STL'],
                                    ['at', '54', 'STL']],
                     'add': [['at', '54', 'LA'], ['unfueled', '54']],
                     'delete': [['at', '54', 'STL'], ['fueled', '54']]}
unit_test = create_filled_action({"?plane": "94", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                 deepcopy(unit_test_action["fuel"]))
assert unit_test == {'action': ['fueled', '94'],
                     'conditions': [['plane', '94'], ['unfueled', '94']],
                     'add': [['fueled', '94']],
                     'delete': [['unfueled', '94']]}


# <a id="continue_state"></a>
# ## continue_state
#
# This function takes the delete and add actions from the action function and applies them to the state to continue on to try and reach the goal **Used by**: [condition_backtracking](#condition_backtracking)
#
# * **current_state**: list: This is the state of the search algorithm that it is currently in.
# * **action**: dict: This is the action that will be used, it should contain values not variables to delete the proper elements from the state.
#
# **returns**: list: this returns the state after it has added or removed anything applicable

# In[14]:


def continue_state(current_state: list, action: dict):
    for act in action["delete"]:
        if act in current_state:
            current_state.remove(act)
    for act in action["add"]:
        current_state.append(act)
    return current_state


# In[15]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
unit_test_state = deepcopy(self_check_state)
unit_test = continue_state(unit_test_state, unit_test_action["fly"])
assert unit_test[-1] == ['unfueled', '?plane']
unit_test_fly = create_filled_action({"?plane": "94", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_test = continue_state(unit_test_state, unit_test_fly)
assert unit_test[-1] == ['unfueled', '94']
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test_state
unit_test_fly = create_filled_action({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fuel"]))
unit_test = continue_state(unit_test_state, unit_test_fly)
assert unit_test[-1] == ['fueled', '97']


# <a id="forward_checking"></a>
# ## forward_checking
#
# The purpose of this function is to take the current expression, the current state, and the preconditions available to use and check to see if the assigned variable is applicable in the state to try and limit how many times the functions recurse by checking the preconditions with the given expression **Uses**: [variable_substitution](#variable_substitution) **Used by**: [future_check](#future_check)
#
# * **expression**: dict: the given expression that is determined how it will be used in the preconditions
# * **current_state**: list: the current state of the search.
# * **preconditions**: list[list]: the list of required conditions for an action
# * **debug**: bool:  This decides if debug information is printed or not.
#
#
# **returns**: bool: returns if the checked expression is applicable with the preconditions

# In[16]:


def forward_checking(expression: dict, current_state: list, preconditions: list[list], debug: bool):
    variable, value = "", ""
    for key in expression:
        value = expression[key]
        variable = key
    for precondition in preconditions:
        if variable in precondition:
            assign_variable = variable_substitution(precondition, expression)
            if assign_variable in current_state:
                return True
    if debug:
        print("Forward checking failed at:", expression)
    return False


# In[17]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
unit_test_fly = create_filled_action({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_preconditions = deepcopy(unit_test_fly["conditions"])
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = forward_checking({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"}, unit_test_state,
                             unit_preconditions, False)
assert unit_test == False
unit_test = forward_checking({"?plane": "97"}, unit_test_state, unit_preconditions, True)
assert unit_test == False
unit_test_action = parse_actions(deepcopy(self_check_actions))
unit_preconditions = deepcopy(unit_test_action["fuel"]["conditions"])
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = forward_checking({"?plane": "97"}, unit_test_state, unit_preconditions, False)
assert unit_test == True


# <a id="future_check"></a>
# ## future_check
#
# This function pretty much is the enabler for forward_checking by making sure that the correct variables are set up as well as doing some booking keeping on the action itself to keep track of the failed expressions so that a repeat of the failed expression is not done **Uses**: [forward_checking](#forward_checking). **Used by**: [condition_backtracking](#condition_backtracking).
#
# * **expression**: dict: the given expression that is determined how it will be used in the preconditions
# * **action**: dict: This is the action that will be used, it should contain values not variables to delete the proper elements from the state.
# * **current_state**: list: the current state of the search.
# * **debug**: bool:  This decides if debug information is printed or not.
#
# **returns**: documentation of the returned value and type.

# In[18]:


def future_check(expression: dict, action: dict, current_state: list, debug: bool):
    fail_check = action["failed"]
    preconditions = deepcopy(action["conditions"])
    if expression in fail_check:
        return False
    if not forward_checking(expression, current_state, preconditions, debug):
        fail_check.append(expression)
        return False
    return True


# In[22]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_fly = create_filled_action({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = future_check({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"}, unit_test_action["fly"],
                         unit_test_state, False)
assert unit_test == False
unit_test = future_check({"?plane": "97"}, unit_test_action["fly"], unit_test_state, True)
assert unit_test == True
unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = future_check({"?plane": "97"}, unit_test_action["fuel"], unit_test_state, False)
assert unit_test == True


# <a id="action_checking"></a>
# ## action_checking
#
# This is used to try and make sure that there is not any repeated conditions in the action that has had its variables replaced, that way there is no redundancy. The main purpose is to make sure that the actions being compared are valid, as well as making sure the current action has not been used in the plan **Uses**: [is_goal](#is_goal) **Used by**: [conditions_backtracking](#condition_backtracking)
#
# * **filled_action**: dict: this is an action that has had its variables replaced with values.
# * **current_state**: list: The current state of the search.
# * **current_plan**: list: the plan on how the current state has been reached.
#
# **returns**: bool: this returns if the action plan is currently valid or not

# In[23]:


def action_checking(filled_action: dict, current_state: list, current_plan: list):
    if not is_goal(filled_action["conditions"], current_state):
        return False
    for element in filled_action["conditions"]:
        if filled_action["conditions"].count(element) > 1:
            return False
    if filled_action["action"] in current_plan:
        return False
    return True


# In[24]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_fly = create_filled_action({"?plane": "1231231", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = action_checking(unit_test_fly, unit_test_state, [])
assert unit_test == False
unit_test = action_checking(unit_test_fly, unit_test_state, [unit_test_fly["action"]])
assert unit_test == False
unit_test_action = parse_actions(deepcopy(self_check_actions))
unit_test_fly = create_filled_action({"?plane": "1231231", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_test = action_checking(unit_test_fly, unit_test_fly["conditions"], [])
assert unit_test == True


# <a id="condition_backtracking"></a>
# ## condition_backtracking
#
# This is one of two backtracking I chose to implement in this algorithm in order for checking the conditions, so that if there were conditions that would not be met to back up and try other routes in a recursive case. This also calls the other recursive backtracking algorithm depending on the current state of the action, state, and plan. If those pass the conditions, it adds a new element to the plan and moves on to the next action, otherwise, it continues on through conditions **Uses**: [create_filled_action](#create_filled_action), [future_check](#future_check), [action_checking](#action_checking), [action_backtracking](#action_backtracking) **Used by**: [action_backtracking](#action_backtracking).
#
# * **current_state**: list: The current state of the search.
# * **goal**: list[list]: The state that we want to have a plan to achieve.
# * **actions**: dict: the full list of possible actions that are being investigated.
# * **current_plan**: list: the plan on how the current state has been reached.
# * **current_action**: dict: the action that is being evaluated for correctness.
# * **debug**: bool:  This decides if debug information is printed or not.
#
# **returns**: bool, list: this returns if it is a valid solution or not as well as the plan to get where it is

# In[57]:


def condition_backtracking(current_state: list, goal: list[list], actions: dict, current_plan: list,
                           current_action: dict, debug: bool):
    for condition in current_action["conditions"]:
        for expression in filter(None, [unification(condition, state_line) for state_line in current_state]):
            if not future_check(expression, actions[current_action["action"][0]], current_state, debug): continue
            filled_action = create_filled_action(expression, deepcopy(current_action))
            if debug:
                print("Conditions that are being checked:", filled_action["conditions"])
            if action_checking(filled_action, current_state, current_plan):
                if debug: print("Action that has been added", filled_action["action"])
                current_plan.append(filled_action["action"])
                return action_backtracking(continue_state(current_state, filled_action), goal, actions, current_plan,
                                           debug)
            else:
                result, filled_action = condition_backtracking(current_state, goal, actions, current_plan,
                                                               filled_action, debug)
                if result: return result, current_plan
    return False, current_plan


# In[26]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_fly = create_filled_action({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = condition_backtracking(unit_test_state, unit_test_state, unit_test_action, [], unit_test_action["fly"],
                                   False)
assert unit_test == (True, [['fly', '1973', 'SFO', 'JFK']])
unit_test = condition_backtracking(unit_test_state[0:5], unit_test_state, unit_test_action, [], unit_test_action["fly"],
                                   False)
assert unit_test == (False, [])
unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_fly = create_filled_action({"?plane": "1973", "?to": "SFO", "?from": "JFK", "?airport": "HR"},
                                     deepcopy(unit_test_action["fly"]))
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = condition_backtracking(unit_test_state, unit_test_state, unit_test_action, [], unit_test_action["fly"],
                                   True)
assert unit_test == (True, [['fly', '1973', 'SFO', 'JFK']])


# <a id="action_backtracking"></a>
# ## action_backtracking
#
# This function is one of two backtracking implementation in this function. This one specific is to backtrack through the actions given that one is unsuccessful or not, to make sure that it is valid and once an action is selected, to call condition_backtracking to continue to find a solution to the goal state that way. **Uses**: (links to functions used). **Used by**: (links to functions used by).
#
# * **current_state**: list: The current state of the search.
# * **goal**: list[list]: The state that we want to have a plan to achieve.
# * **actions**: dict: the full list of possible actions that are being investigated.
# * **current_plan**: list: the plan on how the current state has been reached.
# * **debug**: bool:  This decides if debug information is printed or not.
#
# **returns**: bool, list: this returns if it is a valid solution or not as well as the plan to get where it is

# In[58]:


def action_backtracking(current_state: list, goal: list[list], actions: dict, current_plan: list, debug: bool):
    if is_goal(current_state, goal):
        return True, current_plan
    for action in actions:
        if debug:
            print("Current state:", current_state, "attempted action:", actions[action]["action"])
        result, current_plan = condition_backtracking(deepcopy(current_state), goal, actions, current_plan,
                                                      deepcopy(actions[action]), debug)
        if result:
            return result, current_plan
    return False, current_plan


# In[28]:


unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_fly = create_filled_action({"?plane": "97", "?to": "STL", "?from": "MD", "?airport": "LAM"},
                                     deepcopy(unit_test_action["fly"]))
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = action_backtracking(unit_test_state, unit_test_state, unit_test_action, [], False)
assert unit_test == (True, [])
unit_test = action_backtracking(unit_test_state[3:5], unit_test_state, unit_test_action,
                                [['fly', '1973', 'SFO', 'JFK']], False)
assert unit_test == (True, [['fly', '1973', 'SFO', 'JFK']])
unit_test_action = parse_actions(deepcopy(self_check_actions))
for unit_action in unit_test_action:
    unit_test_action[unit_action]["failed"] = []
unit_test_fly = create_filled_action({"?plane": "1973", "?to": "SFO", "?from": "JFK", "?airport": "HR"},
                                     deepcopy(unit_test_action["fly"]))
unit_test_state = deepcopy(self_check_state)
for i, unit_line in enumerate(unit_test_state):
    unit_test_state[i] = parse(unit_line)
unit_test = action_backtracking(unit_test_state, unit_test_state, unit_test_action, [], True)
assert unit_test == (True, [])


# <a id="forward_planner"></a>
# ## forward_planner
#
# Here is the main function that brings everything together. The main purpose of this function is to gather all of the start state, goal, actions, and if it is being used for debug or not, and call the backtracking algorithm which does infact include forward checking wiht it. One thing that was added here was to add another line to the data, failed, so that when there is a failure within forward checking that it can be checked against to reduce branching and increase efficiency. Once the goal state is reached with back and forward tracking, it returns if the result is valid or not and how it got there. **Uses**:[action_backtracking](#action_backtracking)
#
# * **start_state**: list: this is the start where the plan will be built off of to determine how to reach the goal.
# * **goal**: list[list]: The state that we want to have a plan to achieve.
# * **actions**: dict: the full list of possible actions that are being investigated.
# * * **debug**: bool:  This decides if debug information is printed or not. Default is False
#
# **returns**: list: returns a list of the actions taken to reach the goal state

# In[29]:


def forward_planner(start_state, goal, actions, debug=False):
    copied_state, copied_goal, copied_actions = deepcopy(start_state), deepcopy(goal), deepcopy(actions)
    copied_actions = parse_actions(copied_actions)
    for i, state in enumerate(copied_state):
        copied_state[i] = parse(state)
    for i, line in enumerate(copied_goal):
        copied_goal[i] = parse(line)
    for action in actions:
        copied_actions[action]["failed"] = []
    result, plan = action_backtracking(copied_state, copied_goal, copied_actions, [], debug)
    if not result:
        return plan
    return []



# In[30]:


start_state = [
    "(item Saw)",
    "(item Drill)",
    "(place Home)",
    "(place Store)",
    "(place Bank)",
    "(agent Me)",
    "(at Me Home)",
    "(at Saw Store)",
    "(at Drill Store)"
]

# The goal state:

# In[31]:


goal = [
    "(item Saw)",
    "(item Drill)",
    "(place Home)",
    "(place Store)",
    "(place Bank)",
    "(agent Me)",
    "(at Me Home)",
    "(at Drill Me)",
    "(at Saw Store)"
]

# and the actions/operators:

# In[32]:


actions = {
    "drive": {
        "action": "(drive ?agent ?from ?to)",
        "conditions": [
            "(agent ?agent)",
            "(place ?from)",
            "(place ?to)",
            "(at ?agent ?from)"
        ],
        "add": [
            "(at ?agent ?to)"
        ],
        "delete": [
            "(at ?agent ?from)"
        ]
    },
    "buy": {
        "action": "(buy ?purchaser ?seller ?item)",
        "conditions": [
            "(item ?item)",
            "(place ?seller)",
            "(agent ?purchaser)",
            "(at ?item ?seller)",
            "(at ?purchaser ?seller)"
        ],
        "add": [
            "(at ?item ?purchaser)"
        ],
        "delete": [
            "(at ?item ?seller)"
        ]
    }
}


# In[59]:


plan = forward_planner(start_state, goal, actions)

# In[60]:


for el in plan:
    print(el)
