from pprint import pprint
import random


ALPHABET = "abcdefghijklmnopqrstuvwxyz "


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
assert test == [[15, 'this so is much fun'], [10, 'what do em such dun'], [6, 'wowe so is done fon'],
                [4, 'how did ig ethe rer']]
population = ["thisissomuchfun", "whatdoemsuchdun", "howdidigetherer", "wowesoisdonefon"]
test = evaluate_fitness(population, "guvfvffbzhpusha", ALPHABET, "rot13")
assert test == [[15, 'thisissomuchfun'], [6, 'whatdoemsuchdun'], [2, 'wowesoisdonefon'], [1, 'howdidigetherer']]
population = ["this is so much fun", "what do em such dun", "how did ig ethe rer", "wowe so is done fon"]
test = evaluate_fitness(population, "nuf hcum os si siht", ALPHABET, "reverse")
assert test == [[19, 'this is so much fun'], [10, 'what do em such dun'], [6, 'wowe so is done fon'],
                [4, 'how did ig ethe rer']]


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


def reproduce(mom: str, dad: str, mutation_rate: float, mutation_location: int, mutation_value: str,
              crossover_rate: float, crossover_point: int, random):
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
            mom, dad = select_parents(population, random.randint(0, number_of_population / 2),
                                      random.randint(0, number_of_population / 2))
            son, daughter = reproduce(mom, dad, .25, random.randint(0, size_of_genotype - 1),
                                      alphabet[random.randint(0, len(alphabet) - 1)], .85,
                                      random.randint(0, size_of_genotype), round(random.random(), 2))
            next_population.extend([son, daughter])
        population = next_population
    population = evaluate_fitness(population, target, alphabet, type_of_sort)
    return population[0]



target1 = "this is so much fun"
result1 = genetic_algorithm(target1, ALPHABET, "normal") # do what you need to do for your implementation but don't change the lines above or below.
pprint(result1, compact=True)


target2 = "nuf hcum os si siht"
result2 = genetic_algorithm(target2, ALPHABET, "reverse") # do what you need to do for your implementation but don't change the lines above or below.
pprint(result2, compact=True)

ALPHABET3 = "abcdefghijklmnopqrstuvwxyz"
target3 = "guvfvffbzhpusha"
result3 = genetic_algorithm(target3, ALPHABET3, "rot13") # do what you need to do for your implementation but don't change the lines above or below.
pprint(result3, compact=True)
