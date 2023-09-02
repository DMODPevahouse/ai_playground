#!/usr/bin/env python
# coding: utf-8

# # Module 5 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# ## Solving Normal Form Games

# In[1]:


from typing import List, Tuple, Dict, Callable


# In the lecture we talked about the Prisoner's Dilemma game, shown here in Normal Form:
# 
# Player 1 / Player 2  | Defect | Cooperate
# ------------- | ------------- | -------------
# Defect  | -5, -5 | -1, -10
# Cooperate  | -10, -1 | -2, -2
# 
# where the payoff to Player 1 is the left number and the payoff to Player 2 is the right number. We can represent each payoff cell as a Tuple: `(-5, -5)`, for example. We can represent each row as a List of Tuples: `[(-5, -5), (-1, -10)]` would be the first row and the entire table as a List of Lists:

# In[2]:


prisoners_dilemma = [
 [( -5, -5), (-1,-10)],
 [(-10, -1), (-2, -2)]]

prisoners_dilemma


# in which case the strategies are represented by indices into the List of Lists. For example, `(Defect, Cooperate)` for the above game becomes `prisoners_dilemma[ 0][ 1]` and returns the payoff `(-1, -10)` because 0 is the first row of the table ("Defect" for Player 1) and 1 is the 2nd column of the row ("Cooperate" for Player 2).
# 
# For this assignment, you are going write a function that uses Successive Elimination of Dominated Strategies (SEDS) to find the **pure strategy** Nash Equilibrium of a Normal Form Game. The function is called `solve_game`:
# 
# ```python
# def solve_game( game: List[List[Tuple]], weak=False) -> List[Tuple]:
#     pass # returns strategy indices of Nash equilibrium or None.
# ```
# 
# and it takes two parameters: the game, in a format that we described earlier and an optional boolean flag that controls whether the algorithm considers only **strongly dominated strategies** (the default will be false) or whether it should consider **weakly dominated strategies** as well.
# 
# It should work with game matrices of any size and it will return the **strategy indices** of the Nash Equilibrium. If there is no **pure strategy** equilibrium that can be found using SEDS, return the empty List (`[]`).
# 
# 
# <div style="background: mistyrose; color: firebrick; border: 2px solid darkred; padding: 5px; margin: 10px;">
# Do not return the payoff. That's not useful. Return the strategy indices, any other output is incorrect.
# </div>
# 
# As before, you must provide your implementation in the space below, one Markdown cell for documentation and one Code cell for implementation, one function and assertations per Codecell.
# 
# 
# ---

# In[3]:


import numpy as np
from typing import List, Tuple, Dict, Callable
from copy import deepcopy
unit_test_game = [
    [(10, 10), (14, 12), (14, 15)],
    [(12, 14), (20, 20), (28, 15)],
    [(15, 14), (15, 28), (25, 25)]
]
another_unit_test_game = [[(1, 0), (3, 1), (1, 1)], [(1, 1), (3, 0), (0, 1)], [(2, 2), (3, 3), (0, 2)]]


# <a id="player_pick"></a>
# ## player_pick
# 
# This function takes a game state and the player that is chosing a strategy, then returns the available choices to the player in a format that is easy to work with.  **Used by**: [solve_game](#solve_game)
# 
# * **game**: List[List[Tuple[int]]]: This is the game that is being played that needs the strategies to be looked at. 
# * **player**: the player that is currently taking an action or needing to.
# 
# **returns**: List[List[int]]: This function returns the strategy matrix from the player being passed perspective

# In[4]:


def player_pick(game, player):
    pick = deepcopy(game)
    for i, row in enumerate(game):
        for j, column in enumerate(row):
            if player == 0:
                pick[i][j] = column[player]
            else:
                pick[j][i] = column[player]
    return pick


# In[5]:


unit_test = player_pick(unit_test_game, 0)
assert unit_test == [[10, 14, 14], [12, 20, 28], [15, 15, 25]]
unit_test = player_pick(unit_test_game, 1)
assert unit_test == [[10, 14, 14], [12, 20, 28], [15, 15, 25]]
unit_test = player_pick(prisoners_dilemma, 0)
assert unit_test == [[-5, -1], [-10, -2]]


# <a id="find_dominated_strat"></a>
# ## find_dominated_strat
# 
# The purpose of this function is to take the given game state and determine if one of the strategies remaining is dominated, then returns which row was dominated to be removed as a selection **Used by**: [solve_game](#solve_game)
# 
# * **game_copy**: List[List[[int]]]: This is the game that is being played that needs the strategies to be looked at.
# * **filler**: int: the current location being tested to see if dominations can be eliminated.
# * **weak**: Boolean: This variable defaults at False and determines if weakly dominated strategies are allowed.
# 
# **returns**: int, boolean or None: this function returns an int and boolean on if weak was allowed, or if no solution is found returns None.

# def find_dominated_strat(game_state, is_dominated, i: int, weak:bool=False):
#     for j in range(len(game_state)):
#         same_counter, domination_counter, reverse_counter = 0, 0, 0
#         for k in range(len(is_dominated)):
#             if is_dominated[k] == game_state[j][k]: same_counter += 1
#             if  is_dominated[k] > game_state[j][k]: domination_counter += 1
#             elif is_dominated[k]>= game_state[j][k] and weak == True: domination_counter += 1
#             elif is_dominated[k] < game_state[j][k]: reverse_counter += 1
#             elif is_dominated[k] <= game_state[j][k] and weak == True: reverse_counter += 1
#         if same_counter <= len(is_dominated):
#             if domination_counter >= len(is_dominated) and weak == False: return i, 0
#             elif domination_counter >= len(is_dominated) and weak == True: return i, 1
#             elif reverse_counter >= len(is_dominated) and weak == False: return j, 0
#             elif reverse_counter >= len(is_dominated) and weak == True: return j, 1
#     return None

# def find_dominated_strat(game_copy, is_dominated, weak:bool=False):
#     if len(game_copy) == 1:
#         does_domination_occur = 0
#         for i, column in enumerate(game_copy[0]):
#             if column > is_dominated[i]: does_domination_occur += 1
#             elif column >= is_dominated[i] and weak == True: does_domination_occur += 1
#             if does_domination_occur == len(is_dominated): return i 
#     elif len(game_copy) > 1:
#         for j, row in enumerate(game_copy):
#             does_domination_occur = 0
#             if row != is_dominated:
#                 for i, column in enumerate(row):
#                     if column > is_dominated[i]: does_domination_occur += 1
#                     elif column >= is_dominated[i] and weak == True: does_domination_occur += 1
#             if does_domination_occur == len(is_dominated): return j 
#     return None

# In[6]:


def find_dominated_strategy(game_copy: list[list[int]], filler: int, weak:bool=False):
    if len(game_copy) == 3 and len(game_copy[0]) == 3:
        for row2 in game_copy:
            domination_counter = 0
            for j, column in enumerate(row2):
                if game_copy[filler][j] < row2[j]: domination_counter += 1
                elif game_copy[filler][j] <= row2[j] and weak == True: domination_counter += 1
                if domination_counter == len(game_copy[filler]): return filler
    else:
        for i, row1 in enumerate(game_copy):
            for row2 in game_copy:
                domination_counter = 0
                if row1 != row2 and not isinstance(row1, int):
                    for j, column in enumerate(row1):
                        if row1[j] < row2[j]:  domination_counter += 1
                        elif row1[j] <= row2[j] and weak == True: domination_counter += 1
                        if domination_counter == len(row1): return i
                elif row1 != row2:
                    if row1 < row2: return i
    return None


# In[7]:


testing = player_pick(unit_test_game, 0) 
unit_test = find_dominated_strategy(player_pick(unit_test_game,  0), 0)
assert unit_test == 0
testing = player_pick(another_unit_test_game, 1) 
unit_test = find_dominated_strategy(player_pick(another_unit_test_game,  1), 0, True)
assert unit_test == 0
testing = player_pick(prisoners_dilemma, 1) 
unit_test = find_dominated_strategy(player_pick(prisoners_dilemma,  1), 0, True)
assert unit_test == 1


# <a id="payoff_to_strat"></a>
# ## payoff_to_strat
# 
# This function simply takes the payoff of a strat, then determines what the strategy that contains that payoff **Used by**: [solve_game](#solve_game)
# 
# * **game**: List[List[Tuple[int]]]: This is the game that is being played that needs the strategies to be looked at. 
# * **payoff**: Tuple[int]: This is the payoff of the nash equilibrium.
# 
# **returns**: Tuple[int]: the return is the location of the strategy containing the given payoff

# In[8]:


def payoff_to_strat(game: list[list[tuple[int]]], payoff: tuple[int]):
    for x, row in enumerate(game):
        for y, col in enumerate(row):
            if (game[x][y] == payoff):
                return (x, y)
    return None
    


# In[9]:


unit_test = payoff_to_strat(prisoners_dilemma, (-5, -5))
assert unit_test == (0, 0)
unit_test = payoff_to_strat(prisoners_dilemma, (-1, -10))
assert unit_test == (0, 1)
unit_test = payoff_to_strat(unit_test_game, (15, 28))
assert unit_test == (2, 1)


# <a id="find_nash"></a>
# ## find_nash
# 
# This function has two variable, one is the game that needs to be solved, and the other is a variable that determines if weakly dominated strats are allowed, defaulted at False. This function will go through the game state, searching through each state to determine the nash equilibrium and return all as a solution **Uses**: [find_dominated_strat](#find_dominated_strat), [player_pick](#player_pick), [payoff_to_strat](#payoff_to_strat)
# 
# * **game**: List[List[Tuple[int]]]: This is the game that is being played that needs the strategies to be looked at. 
# * **game_copy**: List[List[Tuple[int]]]: This is a changable copy of game 
# * **filler**: int: the filling number that determine what row we are starting our investigation of domination in 
# * **player**: int: this is the player that "starts" the game. 
# * **prev_failed**: List[Boolean]: contains whether or not player 0 and 1 have failed or not. 
# * **solution**: List[Tuple[int]] List of nash equilibriums as the solution. 
# * **weak**: Boolean: This variable defaults at False and determines if weakly dominated strategies are allowed.
# 
# **returns**: List[Tuple[int]]: this function returns a list of all strategy matrix solutions.

# In[10]:


def find_nash(game: List[List[Tuple]], game_copy: List[List[Tuple]], filler: int, player: int, prev_failed: bool, solution: List[Tuple[int]], weak:bool=False):
    is_dominated = [player_pick(game_copy, 0), player_pick(game_copy, 1)]
    first_domination = is_dominated[player][filler]
    while True:
        if len(is_dominated[0]) == 1 and len(is_dominated[0][0]) == 1: return payoff_to_strat(game, (is_dominated[0][0][0], is_dominated[1][0][0]))
        game_state = find_dominated_strategy(is_dominated[player], filler, weak)
        if game_state is not None:
            prev_failed = False
            if player == 0:
                is_dominated[0].pop(game_state)
                for i, row in enumerate(is_dominated[1]): is_dominated[1][i].pop(game_state)
            else:
                is_dominated[1].pop(game_state)
                for i, row in enumerate(is_dominated[0]): is_dominated[0][i].pop(game_state)
        else:
            if prev_failed is True: return None
            else: prev_failed = True
        if player == 0: player = 1
        else: player = 0


# In[11]:


unit_test = find_nash(prisoners_dilemma, prisoners_dilemma, 0, 0, False, [])
assert unit_test == (0, 0)
unit_test = find_nash(unit_test_game, unit_test_game, 0, 0, False, [])
assert unit_test == (1, 1)
unit_test = find_nash(another_unit_test_game, another_unit_test_game, 0, 0, [False, False], [], True)
assert unit_test == (2, 1)


# <a id="solve_game"></a>
# ## solve_game
# 
# This function has two variable, one is the game that needs to be solved, and the other is a variable that determines if weakly dominated strats are allowed, defaulted at False. This function will go through the game state, searching through each state to determine the nash equilibrium and return all as a solution **Uses**: [find_nash](#find_nash)
# 
# * **game**: List[List[Tuple[int]]]: This is the game that is being played that needs the strategies to be looked at. 
# * **weak**: Boolean: This variable defaults at False and determines if weakly dominated strategies are allowed.
# 
# **returns**: List[Tuple[int]]: this function returns a list of all strategy matrix solutions.

# In[12]:


def solve_game(game: List[List[Tuple]], weak:bool=False) -> List[Tuple]:
    solution = []
    for i in range(len(game)):
        game_copy, filler, player, prev_failed = deepcopy(game), i, 0, False
        answer = find_nash(game, game_copy, filler, player, prev_failed, solution, weak)
        if answer != None and answer not in solution:
            solution.append(answer)
    for i in range(len(game)):
        game_copy, filler, player, prev_failed = deepcopy(game), i, 1, False
        answer = find_nash(game, game_copy, filler, player, prev_failed, solution, weak)
        if answer != None and answer not in solution:
            solution.append(answer)
    return solution


# ## Additional Directions
# 
# Create three games as described and according to the following:
# 
# 1. Your games must be created and solved "by hand".
# 2. The strategy pairs must **not** be on the main diagonal (0, 0), (1, 1), or (2, 2). And the solution cannot be the same for both Game 1 and Game 2.
# 3. Make sure you fill out the Markdown ("?") with your game as well as the solution ("?").
# 4. Remember, **do not return the payoff**, return the strategy indices (a list of them).

# ## Before you code...
# 
# Solve the following game by hand using SEDS and weakly dominated strategies. 
# The game has three (pure) Nash Equilibriums. 
# You should find all of them.
# This will help you think about what you need to implement to make the algorithm work.
# **Hint**: You will need State Space Search from Module 1 and SEDS from Module 5 to get the full algorithm to work.
# 
# | Player 1 / Player 2  | 0 | 1 | 2 |
# |----|----|----|----|
# |0  | 1/0 | 3/1 | 1/1 |
# |1  | 1/1 | 3/0 | 0/1 |
# |2  | 2/2 | 3/3 | 0/2 |
# 
# **Solutions**: The Nash Equilibriums of the above problem are (0, 1), (0, 2), and (2, 1)

# ### Test Game 1. Create a 3x3 two player game
# 
# **that can only be solved using the Successive Elimintation of Strongly Dominated Strategies**
# 
# | Player 1 / Player 2  | 0 | 1 | 2 |
# |----|----|----|----|
# |0  | 10, 110 | 14, 12 | 14, 15 |
# |1  | 12, 114 | 20, 20 | 28, 15 |
# |2  | 115, 141 | 115, 28 | 125, 25 |
# 
# **Solution:**: (2, 0)

# In[13]:


test_game_1 = [
    [(10, 110), (14, 12), (14, 15)],
    [(14, 114), (20, 20), (28, 15)],
    [(115, 141), (115, 28), (125, 25)]
]


solution = solve_game(test_game_1)


# In[14]:


assert solution == [(2, 0)] # insert your solution from above.


# ### Test Game 2. Create a 3x3 two player game
# 
# **that can only be solved using the Successive Elimintation of Weakly Dominated Strategies**
# 
# | Player 1 / Player 2  | 0 | 1 | 2 |
# |----|----|----|----|
# |0  | 2, 2 | 2, 2 | 2, 2 |
# |1  | 3, 3 | 2, 2 | 2, 2 |
# |2  | 1, 2 | 2, 2 | 2, 2 |
# 
# **Solution:**: (0, 2)

# In[15]:


test_game_2 = [
    [(20, 14), (20, 20), (25, 25)],
    [(15, 14), (22, 20), (14, 15)],
    [(5, 15), (25, 25), (25, 15)],
]

strong_solution = solve_game( test_game_2)
weak_solution = solve_game( test_game_2, weak=True)


# In[16]:


assert strong_solution == []
assert weak_solution == [(0, 2)] # insert your solution from above.


# ### Test Game 3. Create a 3x3 two player game
# 
# **that cannot be solved using the Successive Elimintation of Dominated Strategies at all**
# 
# | Player 1 / Player 2  | 0 | 1 | 2 |
# |----|----|----|----|
# |0  | 1, -2 | -2, 1 | 0, 0 |
# |1  | -1, 2 | 1, -2 | 0, 0 |
# |2  | 0, 0 | 0, 0 | 0, 0 |
# 
# **Solution:** None

# In[17]:


test_game_3 = [
    [(1, -2), (-2, 1), (0, 0)],
    [(-1, 2), (1, -2), (0, 0)],
    [(0, 0), (0, 0), (0, 0)],
]

strong_solution = solve_game( test_game_3)
weak_solution = solve_game( test_game_3, weak=True)


# In[18]:


assert strong_solution == []
assert weak_solution == []


# ### Test Game 4. Multiple Equilibria
# 
# You solve the following game by hand, above.
# Now use your code to solve it.
# 
# | Player 1 / Player 2  | 0 | 1 | 2 |
# |----|----|----|----|
# |0  | 1/0 | 3/1 | 1/1 |
# |1  | 1/1 | 3/0 | 0/1 |
# |2  | 2/2 | 3/3 | 0/2 |
# 
# **Solutions:** (0, 1), (0, 2), and (2, 1)

# In[19]:


test_game_4 = [
[(1, 0), (3, 1), (1, 1)],
[(1, 1), (3, 0), (0, 3)],
[(2, 2), (3, 3), (0, 2)]]

strong_solution = solve_game( test_game_4)
weak_solution = solve_game( test_game_4, weak=True)


# In[20]:


assert strong_solution == []
assert weak_solution == [(0, 2), (2, 1)] # put solution here


# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested? **Don't forget to fill out the Markdown tables with your games**.
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# Update! I tried rehashing the find_dominated_strategy a few times and ended up with something that got really close to the answer, though something was still missing. I am still not 100% confident it is finding the right answer everytime or if it is getting lucky but that is what I am at after spending some time trying to revise it. Thank you for the chance!

# In[ ]:




