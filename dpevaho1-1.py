#!/usr/bin/env python
# coding: utf-8

# # Module 2 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# In[1]:


from pprint import pprint


# ## Local Search - Genetic Algorithm
# 
# There are some key ideas in the Genetic Algorithm.
# 
# First, there is a problem of some kind that either *is* an optimization problem or the solution can be expressed in terms of an optimization problem.
# For example, if we wanted to minimize the function
# 
# $$f(x) = \sum (x_i - 0.5)^2$$
# 
# where $n = 10$.
# This *is* an optimization problem. Normally, optimization problems are much, much harder.
# 
# ![Eggholder](http://www.sfu.ca/~ssurjano/egg.png)!
# 
# The function we wish to optimize is often called the **objective function**.
# The objective function is closely related to the **fitness** function in the GA.
# If we have a **maximization** problem, then we can use the objective function directly as a fitness function.
# If we have a **minimization** problem, then we need to convert the objective function into a suitable fitness function, since fitness functions must always mean "more is better".
# 
# Second, we need to *encode* candidate solutions using an "alphabet" analogous to G, A, T, C in DNA.
# This encoding can be quite abstract.
# You saw this in the Self Check.
# There a floating point number was encoded as bits, just as in a computer and a sophisticated decoding scheme was then required.
# 
# Sometimes, the encoding need not be very complicated at all.
# For example, in the real-valued GA, discussed in the Lectures, we could represent 2.73 as....2.73.
# This is similarly true for a string matching problem.
# We *could* encode "a" as "a", 97, or '01100001'.
# And then "hello" would be:
# 
# ```
# ["h", "e", "l", "l", "o"]
# ```
# 
# or
# 
# ```
# [104, 101, 108, 108, 111]
# ```
# 
# or
# 
# ```
# 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1
# ```
# 
# In Genetics terminology, this is the **chromosome** of the individual. And if this individual had the **phenotype** "h" for the first character then they would have the **genotype** for "h" (either as "h", 104, or 01101000).
# 
# To keep it straight, think **geno**type is **genes** and **pheno**type is **phenomenon**, the actual thing that the genes express.
# So while we might encode a number as 10110110 (genotype), the number itself, 182, is what goes into the fitness function.
# The environment operates on zebras, not the genes for stripes.

# ## String Matching
# 
# You are going to write a Genetic Algorithm that will solve the problem of matching a target string (at least at the start).
# Now, this is kind of silly because in order for this to work, you need to know the target string and if you know the target string, why are you trying to do it?
# Well, the problem is *pedagogical*.
# It's a fun way of visualizing the GA at work, because as the GA finds better and better candidates, they make more and more sense.
# 
# Now, string matching is not *directly* an optimization problem so this falls under the general category of "if we convert the problem into an optimization problem we can solve it with an optimization algorithm" approach to problem solving.
# This happens all the time.
# We have a problem.
# We can't solve it.
# We convert it to a problem we *can* solve.
# In this case, we're using the GA to solve the optimization part.
# 
# And all we need is some sort of measure of the difference between two strings.
# We can use that measure as a **loss function**.
# A loss function gives us a score tells us how similar two strings are.
# The loss function becomes our objective function and we use the GA to minimize it by converting the objective function to a fitness function.
# So that's the first step, come up with the loss/objective function.
# The only stipulation is that it must calculate the score based on element to element (character to character) comparisons with no global transformations of the candidate or target strings.
# 
# And since this is a GA, we need a **genotype**.
# The genotype for this problem is a list of "characters" (individual letters aren't special in Python like they are in some other languages):
# 
# ```
# ["h", "e", "l", "l", "o"]
# ```
# 
# and the **phenotype** is the resulting string:
# 
# ```
# "hello"
# ```
# 
# In addition to the generic code and problem specific loss function, you'll need to pick parameters for the run.
# These parameters include:
# 
# 1. population size
# 2. number of generations
# 3. probability of crossover
# 4. probability of mutation
# 
# You will also need to pick a selection algorithm, either roulette wheel or tournament selection.
# In the later case, you will need a tournament size.
# This is all part of the problem.
# 
# Every **ten** (10) generations, you should print out the fitness, genotype, and phenotype of the best individual in the population for the specific generation.
# The function should return the best individual *of the entire run*, using the same format.

# In[2]:


ALPHABET = "abcdefghijklmnopqrstuvwxyz "


# In[3]:


import random


# <a id="determine_reverse_fitness"></a>
# ## determine_reverse_fitness
# 
# This function would be called specifically by evaluate_fitness when it is wanting to determine the fitness of the string in reverse order. The function achieves that by essentially comparing the gene in the target to the last gene in the current spot of the algorithm. So index 0 would compare against index -1, which in python is the last element. As the index climbs for the target, the index shrinks at the same rate. **Used by** [evaluate_fitness](#evaluate_fitness)
# 
# * **target: str**: This is the goal of the string, in this case it is rot13, or rotated to a character 13 steps away.
# * **current: str**: This is where the current state of the algorithm is.
# 
# **returns**: int: the fitness level based on how many characters match the target.

# In[4]:


def determine_reverse_fitness(target: str, current: str):
    fit = 0
    for i, gene in enumerate(target):
        if gene == current[-i - 1]:
            fit += 1
    return fit


# In[5]:


test = determine_reverse_fitness("nuf hcum os si siht", "this is so much fun")
assert test == 19
test = determine_reverse_fitness("wow", "wpw")
assert test == 2
test = determine_reverse_fitness("esrever si eman ym", "my name is reverse")
assert test == 18


# <a id="determine_rot13_fitness"></a>
# ## determine_rot13_fitness
# 
# This function is the fitness function to determine the fitness in a rot13 manner. It performs this action by checking the current state against the target, but rotated 13, rot13. It handles this by adding 13 to the index of the value, and in order to prevent going over index, when the number is larger then 26, it subtract the altered index by 26 to find the proper solution **Used by** [evaluate_fitness](#evaluate_fitness)
# 
# * **target: str**: This is the goal of the string, in this case it is reverse order.
# * **current: str**: This is where the current state of the algorithm is.
# 
# **returns**: int: the fitness level based on how many characters match the target.

# In[6]:


def determine_rot13_fitness(target: str, current: str, alphabet: str):
    fit = 0
    for i, gene in enumerate(target):
        altered_gene = alphabet.find(gene) + 13
        if altered_gene > 25:
            altered_gene = altered_gene - 26
        rot13_target = alphabet[altered_gene]
        if rot13_target == current[i]:
            fit += 1
    return fit


# In[7]:


ALPHABET3 = "abcdefghijklmnopqrstuvwxyz"
test = determine_rot13_fitness("guvfvffbzhpusha", "thisissomuchfun", ALPHABET3)
assert test == 15
test = determine_rot13_fitness("guvfvffbzhpusha", "zzzzzqqqqqqzzxx", ALPHABET3)
assert test == 0
test = determine_rot13_fitness("guvfvffbzhpusha", "thiszqsoqqqzfun", ALPHABET3)
assert test == 9


# <a id="determine_fitness"></a>
# ## determine_fitness
# 
# This function is the basic fitness function. It determines the level of fitness of an string compared to the target string. Objects that are not strings can also be compared. This is performed by taking the current state and comparing each index with the target string to determine its fitness level. The fitness function is the main way of determining if the current state is objectively close to the target state. It is used to compare and determine the best parents. **Used by** [evaluate_fitness](#evaluate_fitness)
# 
# * **target: str**: This is the goal of the string that the algorithm is trying to match.
# * **current: str**: This is where the current state of the algorithm is.
# 
# **returns**: int: the fitness level based on how many characters match the target.

# In[8]:


def determine_fitness(target: list[str], current: list[str]):
    fit = 0
    for i, gene in enumerate(target):
        if gene == current[i]:
            fit += 1
    return fit


# In[9]:


fit = determine_fitness("this is so much fun", "this is so much fun")
assert fit == 19
fit = determine_fitness("Does the cow fly", "Toes sam dog cry")
assert fit == 8 
fit = determine_fitness("iwantnoletterthesame", "medoallzzzzzzzzzzzzqqqqqq")
assert fit == 0


# <a id="crossover"></a>
# ## crossover
# 
# The purpose of this function is to take two parent strings and cross their genetypes. In order for this to happen, there is a crossover_rate and a random provided value. If the value is less then the crossover_rate, the crossover happens at the determined location, otherwise no change. **Used by**: [reproduce](#reproduce)
# 
# * **mom**: the first parent used to determine the chromosome of the children.
# * **dad**: the second parent used to determine the chromosome of the children.
# * **crossover_rate**: The percent chance of a crossover happening.
# * **random**: a value past to the function that will be created in a random fashion, but is used this way for successful unit tests.
# * **crossover_point**: this is the location of a crossover happening, if it happens.
# 
# **returns**: str, str: This function returns either the original mom, dad if no crossover happens, or a son, daughter, if the crossover does happen

# In[10]:


def crossover(mom: str, dad: str, crossover_rate: float, random: float, crossover_point: int):
    if random < crossover_rate:
        son = mom[:crossover_point] + dad[crossover_point:]
        daughter = dad[:crossover_point] + mom[crossover_point:]
        return son, daughter
    else:
        return mom, dad


# In[11]:


test1, test2 = crossover("iammom", "dadami", .85, .50, 3)
assert test1 == "iamami"
assert test2 == "dadmom"
test1, test2 = crossover("mynameistestcase", "names are not real", .85, .84, 5)
assert test1 == "mynam are not real"
assert test2 == "nameseistestcase"
test1, test2 = crossover("this does not matter", "crossover fails", .85, .85, 7)
assert test1 == "this does not matter"
assert test2 == "crossover fails"


# <a id="mutation"></a>
# ## mutation
# 
# This function is used to have a small chance of changing the children through a process of mutation. It recieves a random float variable that is created when it is called, which was done intentionally to be able to properly test this function. If the random float is smaller then the mutation_point then it uses another random position given in the same way to change to a random later. **Used by**: [reproduce](#reproduce)
# 
# * **son**: The first child that will be mutated or not.
# * **daughter**: the second child that will be mutated or not.
# * **mutation_rate**: The percent chance of a mutation happening.
# * **random**: a value past to the function that will be created in a random fashion, but is used this way for successful unit tests.
# * **mutation_location**: the location that a mutation would occur.
# * **mutation_value**: If a mutation occurs, the value that would be changed.
# 
# **returns**: str, str: this function returns the children given to it, and they will either be mutated or not mutated.

# In[12]:


def mutation(son: str, daughter: str, mutation_rate: float, random: float, mutation_location: int, mutation_value: str):
    if random < mutation_rate:
        son = son[:mutation_location] + mutation_value + son[mutation_location + 1:]
        daughter = daughter[:mutation_location] + mutation_value + daughter[mutation_location + 1:]
        return son, daughter
    else:
        return son, daughter
        


# In[13]:


test1, test2 = mutation("mutateme", "memutate", .05, .03, 4, "z")
assert test1 == "mutazeme"
assert test2 == "memuzate"
test1, test2 = mutation("dontmutateme", "memutatedont", .05, .05, 4, "z")
assert test1 == "dontmutateme"
assert test2 == "memutatedont"
test1, test2 = mutation("whyhaveibeenmutated", "mutatedbeenihavewhy", .05, .01, 7, "u")
assert test1 == "whyhaveubeenmutated"
assert test2 == "mutatedueenihavewhy"


# <a id="population_creation"></a>
# ## population_creation
# 
# The purpose of this function was that the goal would be to have as random as a population as possible for genetic diversity, but also a consistently testable function. In order to reach that, a simple function was created to create the population. **Used by**: [genetic_algorithm](#genetic_algorithm)
# 
# * **child**: this is the string that will be past to the function, it is going to continue to grow until it reaches the same size as the target.
# * **gene**: the specific gene added to the chromosome until it reaches the same length of the .
# 
# **returns**: str: the chromosome that is being created.

# In[14]:


def population_creation(child: str, gene: str):
    new_child = child + gene
    return new_child
    


# In[15]:


test = population_creation("meis", "u")
assert test == "meisu"
test = population_creation("", "i")
assert test == "i"
test = population_creation("hereisa long string", " ")
assert test == "hereisa long string "


# <a id="select_parents"></a>
# ## select_parents
# 
# To select parents in this implementation of genetic_algorithm, tournament style is used. What will be passed to it is a random value, however the random value is only looking at the top half of the population. That is a decent amount to still encourage genetic diversity, while throwing away values that would only muck up the gene pool **Used by**: [genetic_algorithm](#genetic_algorithm)
# 
# * **population**: The current population used to determine the parents.
# * **random_mom**: a randomly selected first parent for reproduction.
# * **random_dad**: a randomly selected second parent for reproduction.
# 
# **returns**: str, str: This returns two parents that will be used for reproduction of the gene pool.

# In[16]:


def select_parents(population: list[list[int, str]], random_mom: int, random_dad: int):
    mom = population[random_mom][1]
    dad = population[random_dad][1]
    return mom, dad


# In[17]:


test = select_parents([[12, "iamme"], [9, "meiam"], [8, "whyis"], [7, "iswhy"]], 1, 0)
assert test == ('meiam', 'iamme')
test = select_parents([[12, "iamme"], [9, "meiam"], [8, "whyis"], [7, "iswhy"]], 2, 3)
assert test == ('whyis', 'iswhy')
test = select_parents([[12, "iamme"], [9, "meiam"], [8, "whyis"], [7, "iswhy"]], 0, 3)
assert test == ('iamme', 'iswhy')


# <a id="evaluate_fitness"></a>
# ## evaluate_fitness
# 
# This function is the starting place to evaluate fitness of the population. It takes in the target, population, characterization, and the type of sort then uses that information to distribute to the needed fitness functions for the operations that are needed to assigne each fitness value to the proper member of the population. After that happens, it sorts the data based on the highest level of fitness so that later on that value can be used to determine parents with the highest fitness values. **Uses**: [determine_fitness](#determine_fitness), [determine_reverse_fitness](#determine_reverse_fitness), [determine_rot13_fitness](#determine_rot13_fitness). **Used by**: [genetic_algorithm](#genetic_algorithm).
# 
# * **population**: The population whose fitness is ready to be evaluated.
# * **target**: The target that is being compared to determine fitness.
# * **alphabet**: The characters that are used in the object being compared.
# * **type_of_sort**: This string determines the type of fitness function that will be used, if no value is presented it will default to a normal fitness function.
# 
# **returns**: list[int, str]: The return is a population who each chromosome in the population has been assigned a fitness value

# In[18]:


def evaluate_fitness(population: list[str], target: str, alphabet: str, type_of_sort="normal"):
    new_population = []
    for person in population:
        if type_of_sort == "rot13":
            new_population.append([determine_rot13_fitness(target, person, alphabet), person])
        elif type_of_sort == "reverse":
            new_population.append([determine_reverse_fitness(target, person), person])
        else:
            new_population.append([determine_fitness(target, person), person])
    new_population.sort(key=lambda x: x[0], reverse=True)
    return new_population


# In[19]:


population = ["this so is much fun", "what do em such dun", "how did ig ethe rer", "wowe so is done fon"]
test = evaluate_fitness(population, "this is so much fun", ALPHABET)
assert test == [[15, 'this so is much fun'], [10, 'what do em such dun'], [6, 'wowe so is done fon'], [4, 'how did ig ethe rer']]
population = ["thisissomuchfun", "whatdoemsuchdun", "howdidigetherer", "wowesoisdonefon"]
test = evaluate_fitness(population, "guvfvffbzhpusha", ALPHABET, "rot13")
assert test == [[15, 'thisissomuchfun'], [6, 'whatdoemsuchdun'], [2, 'wowesoisdonefon'], [1, 'howdidigetherer']]
population = ["this is so much fun", "what do em such dun", "how did ig ethe rer", "wowe so is done fon"]
test = evaluate_fitness(population, "nuf hcum os si siht", ALPHABET, "reverse")
assert test == [[19, 'this is so much fun'], [10, 'what do em such dun'], [6, 'wowe so is done fon'], [4, 'how did ig ethe rer']]


# <a id="reproduce"></a>
# ## reproduce
# 
# This function will take in two parents previously selected in genetic_algorithm then use those genotypes to determine the genotypes of the children. The parents will be passed into the crossover function to determine if crossover will be used and where based on values given, then after then head straight to the mutation function to determine if a mutation will be used and how it will be used based on random data that the genetic_algorithm will pass. The reason it is not passed here is for testing purposes. **Uses**: [crossover](#crossover), [mutation](#mutation). **Used by**: [genetic_algorithm](#genetic_algorithm).
# 
# * **mom**: the first parent used to determine the chromosome of the children.
# * **dad**: the second parent used to determine the chromosome of the children.
# * **mutation_rate**: The percent chance of a mutation happening.
# * **mutation_location**: the location that a mutation would occur.
# * **mutation_value**: If a mutation occurs, the value that would be changed.
# * **crossover_rate**: The percent chance of a crossover happening.
# * **crossover_point**: this is the location of a crossover happening, if it happens.
# * **random**: a value past to the function that will be created in a random fashion, but is used this way for successful unit tests.
# 
# **returns**: str, str: This function returns the children that will be created from the selected parents after they have gone through a crossover and mutation process

# In[20]:


def reproduce(mom: str, dad: str, mutation_rate: float, mutation_location: int, mutation_value: str, crossover_rate: float, crossover_point: int, random):
    son, daughter = crossover(mom, dad, crossover_rate, random, crossover_point)
    son, daughter = mutation(son, daughter, mutation_rate, random, mutation_location, mutation_value)
    return son, daughter


# In[21]:


test1, test2 = reproduce("mynameiswhat", "whatismyname", .05, 4, "q", .85, 6, .03)
assert test1 == "mynaqemyname"
assert test2 == "whatqsiswhat"
test1, test2 = reproduce("nomutationpls", "plsnomutation", .05, 5, "z", .85, 5, .90)
assert test1 == "nomutationpls"
assert test2 == "plsnomutation"
test1, test2 = reproduce("onlycrossoverpls", "plsonlycrossover", .05, 4, "q", .85, 6, .53)
assert test1 == "onlycrycrossover"
assert test2 == "plsonlossoverpls"


# <a id="genetic_algorithm"></a>
# ### genetic_algorithm
#  This function is a algorithm in the overall category of evolution algorithms. It is based on the basics of biology of parents having traits be given to children, chromosomes/genotypes, with the converted being phenotypes. This specific case though, the genotype and phenotype is the same. The objective of the function is to create a random population of similar length strings with random characters. After that, the function will use the type of sort being used, in this case normal, reverse, or rot13 to see the fitness levels as a starting point and then start reproducing. Through a process of using crossover and mutation, while determining and using fitness levels to pass down traits that lead to higher fitness levels. In this case tournament selection is used by reproduce to select parents in the top half of fitness levels so that a level of genetic diversity exists while removing those who will bring down the genepool. There is also a source of mutation that will occur to further increase diversity to try and reach the answer. From here, new generations are created in the population to continually get closer to the target, and often time reaching the target. Finally, after 350 generations in this example, found by experimentation, the algorithm will end and provide the member of the population that has the highest fitness. To see the progression, the algorithm also prints out the top member every 10 generations nit test th[population_creation](#population_creation), [evaluate_fitness](#evaluate_fitness), [select_parents](#select_parents), [reproduce](#reproduce)tno* **target**: The target that is being compared to determine fitness.
# * **alphabet**: The characters that are used in the object being compared.
# * **type_of_sort**: This string determines the type of fitness function that will be used, if no value is presented it will default to a normal fitness function.
#  has a lot**: list[int, str]: This function returns a list that contains the fitness number of how close the state is to the target, and the string attached to said fitness number parameters We could reduce those by using a Dictionary.
# 2. There are a lot of different possible behaviors, including problem specific behaviors. We could use higher order functions.
# 
# Beyond these hints, I leave those decisions to you.
# 
# 
# *This is very Mission Impossible. After reading the directions in this Markdown cell, when the time is right, remove them  (everything between and including the parentheses) and replace with your documentation for `genetic_algorithm`! I have started you off.*
# 
# **)**
# 
# 

# In[22]:


def genetic_algorithm(target: str, alphabet: str, type_of_sort="normal"):
    number_of_population, size_of_genotype, population = 50, len(target), []
    for i in range(number_of_population):
        genotype = ""
        for j in range(size_of_genotype):
            genotype = population_creation(genotype, alphabet[random.randint(0, len(alphabet) - 1)])
        population.append(genotype)
    for generations in range(350):
        population = evaluate_fitness(population, target, alphabet, type_of_sort)
        next_population = []
        if generations % 10 == 0:
            print(population[0])
        for generation in range(1, number_of_population, 2):
            mom, dad = select_parents(population, random.randint(0, number_of_population / 2), random.randint(0, number_of_population / 2 ))
            son, daughter = reproduce(mom, dad, .25, random.randint(0, size_of_genotype - 1), alphabet[random.randint(0, len(alphabet) - 1)], .85, random.randint(0, size_of_genotype), round(random.random(), 2))
            next_population.extend([son, daughter])
        population = next_population
    population = evaluate_fitness(population, target, alphabet, type_of_sort)
    return population[0]


# ## Problem 1
# 
# The target is the string "this is so much fun".
# The challenge, aside from implementing the basic algorithm, is deriving a fitness function based on "b" - "p" (for example).
# The fitness function should come up with a fitness score based on element to element comparisons between target v. phenotype.

# In[23]:


target1 = "this is so much fun"


# In[24]:


# set up if you need it.


# In[25]:


result1 = genetic_algorithm(target1, ALPHABET, "normal") # do what you need to do for your implementation but don't change the lines above or below.


# In[26]:


pprint(result1, compact=True)


# ## Problem 2
# 
# You should have working code now.
# The goal here is to think a bit more about fitness functions.
# The target string is now, 'nuf hcum os si siht'.
# This is obviously target #1 but reversed.
# If we just wanted to match the string, this would be trivial.
# Instead, this problem, we want to "decode" the string so that the best individual displays the target forwards.
# In order to do this, you'll need to come up with a fitness function that measures how successful candidates are towards this goal.
# The constraint is that you may not perform any global operations on the target or individuals.
# Your fitness function must still compare a single gene against a single gene.
# Your solution will likely not be Pythonic but use indexing.
# That's ok.
# <div style="background: lemonchiffon; margin:20px; padding: 20px;">
#     <strong>Important</strong>
#     <p>
#         You may not reverse an entire string (either target or candidate) at any time.
#         Everything must be a computation of one gene against one gene (one letter against one letter).
#         Failure to follow these directions will result in 0 points for the problem.
#     </p>
# </div>
# 
# The best individual in the population is the one who expresses this string *forwards*.

# In[27]:


target2 = "nuf hcum os si siht"


# In[28]:


# set up if you need it.


# In[29]:


result2 = genetic_algorithm(target2, ALPHABET, "reverse") # do what you need to do for your implementation but don't change the lines above or below.


# In[30]:


pprint(result2, compact=True)


# ## Problem 3
# 
# This is a variation on the theme of Problem 2.
# The Caeser Cypher replaces each letter of a string with the letter 13 characters down alphabet (rotating from "z" back to "a" as needed).
# This is also known as ROT13 (for "rotate 13").
# Latin did not have spaces (and the space is not continguous with the letters a-z) so we'll remove them from our alphabet.
# Again, the goal is to derive a fitness function that compares a single gene against a single gene, without global transformations.
# This fitness function assigns higher scores to individuals that correctly decode the target.
# 
# <div style="background: lemonchiffon; margin:20px; padding: 20px;">
#     <strong>Important</strong>
#     <p>
#         You may not apply ROT13 to an entire string (either target or candidate) at any time.
#         Everything must be a computation of one gene against one gene.
#         Failure to follow these directions will result in 0 points for the problem.
#     </p>
# </div>
# 
# The best individual will express the target *decoded*.

# In[31]:


ALPHABET3 = "abcdefghijklmnopqrstuvwxyz"


# In[32]:


target3 = "guvfvffbzhpusha"


# In[33]:


# set up if you need it


# In[34]:


result3 = genetic_algorithm(target3, ALPHABET3, "rot13") # do what you need to do for your implementation but don't change the lines above or below.


# In[35]:


pprint(result3, compact=True)


# ## Problem 4
# 
# There is no code for this problem.
# 
# In Problem 3, we assumed we knew what the shift was in ROT-13.
# What if we didn't?
# Describe how you might solve that problem including a description of the solution encoding (chromosome and interpretation) and fitness function. Assume we can add spaces into the message.

# If spaces were included in this I think it would not really change to much, it would essentially just change a letter to the alphabet so there would be 27 letters instead of the normal 26.
# 
# Now solving the problem with no knowledge of the encryption type, that sounds a lot more difficult to me. In once sense, something that could increase the cost and space complexity by a lot is by instead of running the determine_fitness as I did in my solution (specifically for rot13), an implementation could be for each known character, in the case of this problems it would be the lowercase alphabet plus a space, and running the determine fitness and program for every single possible rotation, from 1-26. That would vastly increase the complexity but it would produce a solution for every single possible rotation, allowing the user to compare what would make more sense at the end. This would assume that there is a moderate amount of trust in the algorithm already in order to solve it.
# 
# That would essentially mean running the entire algorithm 26 times to account for every rotation, and as strings get longer (depending on what is getting decoded) would be very expensive. That can be done by either making the fitness function for the chromosomes flexible and able to change, or by doing it all every time and giving the answer for every single one. That would be costly.
# 
# There could also be the thought of taking a small sample of the chromosome encoding and seeing if there was an answer in that sample by trying all rotations. Once one was found, continue to the rest. That sounds complicated in code to me, but would be significantly cheaper. 
# 
# These changes would happen in the fitness function, as that is what is determining how close a string, in this case, is to another string, and the encoding would be figured out here. That is because, since it determines letter fitness, it would be able to tell the algorithm which is the most likely answer. 
# 
# I think the way I would like to think I would go about this is to run a multithreaded test on here, where I run all 26 possibilities at the same time, get a result from each one and determine which result would be the most likely to be the correct encoding. That would still require human intervention, and the only way I can think of that would not require human intervention would be to use a large language model to analyze the answers to determine with an objective optimization what would be the most likely answer to the problem. 
# 

# ## Challenge
# 
# **You do not need to do this problem and it won't be graded if you do. It's just here if you want to push your understanding.**
# 
# The original GA used binary encodings for everything.
# We're basically using a Base 27 encoding.
# You could, however, write a version of the algorithm that uses an 8 bit encoding for each letter (ignore spaces as they're a bit of a bother).
# That is, a 4 letter candidate looks like this:
# 
# ```
# 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1
# ```
# 
# If you wrote your `genetic_algorithm` code general enough, with higher order functions, you should be able to implement it using bit strings instead of latin strings.

# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.
