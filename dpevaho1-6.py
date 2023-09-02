#!/usr/bin/env python
# coding: utf-8

# # Module 7 - Programming Assignment
# 
# ## Directions
# 
# 1. Change the name of this file to be your JHED id as in `jsmith299.ipynb`. Because sure you use your JHED ID (it's made out of your name and not your student id which is just letters and numbers).
# 2. Make sure the notebook you submit is cleanly and fully executed. I do not grade unexecuted notebooks.
# 3. Submit your notebook back in Blackboard where you downloaded this file.
# 
# *Provide the output **exactly** as requested*

# # Unification
# 
# This is actually Part I of a two part assignment. In a later module, you'll implement a Forward Planner. In order to do that, however, you need to have a unifier. It is important to note that you *only* need to implement a unifier. Although the module talked about resolution, you do not need to implement anything like "standardizing apart". From the unifier's point of view, that should already have been done.
# 
# Unification is simply the *syntactic* balancing of expressions. There are only 3 kinds of expressions: constants, lists and (logic) variables. Constants and lists are only equal to each other if they're exactly the same thing or can be made to be the same thing by *binding* a value to a variable.
# 
# It really is that simple...expressions must be literally the same (identical) except if one or the other (or both) has a variable in that "spot".
# 
# ## S-Expressions
# 
# With that out of the way, we need a language with which to express our constants, variables and predicates and that language will be based on s-expressions.
# 
# **constants** - There are two types of constants, values and predicates. Values should start with an uppercase letter. Fred is a constant value, so is Barney and Food. Predicates are named using lowercase letters. loves is a predicate and so is hates. This is only a convention. Secret: your code does not need to treat these two types of constants differently.
# 
# **variables** - these are named using lowercase letters but always start with a question mark. ?x is a variable and so is ?yum. This is not a convention.
# 
# **expressions (lists)** - these use the S-expression syntax a la LISP. (loves Fred Wilma) is an expression as is (friend-of Barney Fred) and (loves ?x ?y).
# 
# ## Parsing
# 
# These functions are already included in the starter .py file.

# In[1]:


import tokenize
from io import StringIO


# This uses the above libraries to build a Lisp structure based on atoms. It is adapted from [simple iterator parser](http://effbot.org/zone/simple-iterator-parser.htm). The first function is the `atom` function.

# In[2]:


def atom( next, token):
    if token[ 1] == '(':
        out = []
        token = next()
        while token[ 1] != ')':
            out.append( atom( next, token))
            token = next()
            if token[ 1] == ' ':
                token = next()
        return out
    elif token[ 1] == '?':
        token = next()
        return "?" + token[ 1]
    else:
        return token[ 1]


# The next function is the actual `parse` function:

# In[3]:


def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())


# **Note** there was a change between 2.7 and 3.0 that "hid" the next() function in the tokenizer.

# From a Python perspective, we want to turn something like "(loves Fred ?x)" to ["loves" "Fred" "?x"] and then work with the second representation as a list of strings. The strings then have the syntactic meaning we gave them previously.

# In[4]:


parse("Fred")


# In[5]:


parse( "?x")


# In[6]:


parse( "(loves Fred ?x)")


# In[7]:


parse( "(father_of Barney (son_of Barney))")


# ## Unifier
# 
# Now that that's out of the way, here is the imperative pseudocode for unification. This is a classic recursive program with a number of base cases. Students for some reason don't like it, try the algorithm in the book, can't get it to work and then come back to this pseudocode.
# 
# Work through the algorithm by hand with your Self-Check examples if you need to but I'd suggest sticking with this implementation. It does work.
# 
# Here is imperative pseudocode for the algorithm:
# 
# ```
# def unification( exp1, exp2):
#     # base cases
#     if exp1 and exp2 are constants or the empty list:
#         if exp1 = exp2 then return {}
#         else return FAIL
#     if exp1 is a variable:
#         if exp1 occurs in exp2 then return FAIL
#         else return {exp1/exp2}
#     if exp2 is a variable:
#         if exp2 occurs in exp1 then return FAIL
#         else return {exp2/exp1}
# 
#     # inductive step
#     first1 = first element of exp1
#     first2 = first element of exp2
#     result1 = unification( first1, first2)
#     if result1 = FAIL then return FAIL
#     apply result1 to rest of exp1 and exp2
#     result2 = unification( rest of exp1, rest of exp2)
#     if result2 = FAIL then return FAIL
#     return composition of result1 and result2
# ```
# 
# `unification` can return...
# 
# 1. `None` (if unification completely fails)
# 2. `{}` (the empty substitution list) or 
# 3. a substitution list that has variables as keys and substituted values as values, like {"?x": "Fred"}. 
# 
# Note that the middle case sometimes confuses people..."Sam" unifying with "Sam" is not a failure so you return {} because there were no variables so there were no substitutions. You do not need to further resolve variables. If a variable resolves to an expression that contains a variable, you don't need to do the substition.
# 
# If you think of a typical database table, there is a column, row and value. This Tuple is a *relation* and in some uses of unification, the "thing" in the first spot..."love" above is called the relation. If you have a table of users with user_id, username and the value then the relation is:
# 
# `(login ?user_id ?username)`
# 
# *most* of the time, the relation name is specified. But it's not impossible for the relation name to be represented by a variable:
# 
# `(?relation 12345 "smooth_operator")`
# 
# Your code should handle this case (the pseudocode does handle this case so all  you have to do is not futz with it).
# 
# Our type system is very simple. We can get by with just a few boolean functions. The first tests to see if an expression is a variable.

# In[8]:


def is_variable( exp):
    return isinstance( exp, str) and exp[ 0] == "?"


# In[9]:


is_variable( "Fred")


# In[10]:


is_variable( "?fred")


# The second tests to see if an expression is a constant:

# In[11]:


def is_constant( exp):
    return isinstance( exp, str) and not is_variable( exp)


# In[12]:


is_constant( "Fred")


# In[13]:


is_constant( "?fred")


# In[14]:


is_constant( ["loves", "Fred", "?wife"])


# It might also be useful to know that:
# 
# <code>
# type( "a")
# &lt;type 'str'>
# type( "a") == str
# True
# type( "a") == list
# False
# type( ["a"]) == list
# True
# </code>
# 
# 
# You need to write the `unification` function described above. It should work with two expressions of the type returned by `parse`. See `unify` for how it will be called. It should return the result of unification for the two expressions as detailed above and in the book. It does not have to make all the necessary substitions (for example, if ?y is bound to ?x and 1 is bound to ?y, ?x doesn't have to be replaced everywhere with 1. It's enough to return {"?x":"?y", "?y":1}. For an actual application, you would need to fix this!)
# 
# -----

# <a id="variable_expression_check"></a>
# ## variable_expression_check
# 
# This function takes in the two expressions being looked into in unification, assuming they contain a variable, then makes sure they are not the same variable. Should that fail, it will fail the whole function, otherwise, it will assign the variable to a dictionary with that assignment **Used by**: [unification](#unification)
# 
# * **exp1**: str: This is the first expression being looked at, which is the expression the function will assign a variable to.
# * **exp2**: str: The second expression, what the variable will be assigned to should it pass.
# * **answer**: dict[str: str] the dictionary that contains all the assignments for the expressions.
# 
# **returns**: None or dict[str: str]: the function returns either none if it fails, or the dict that was passed to it, except with an additional entry. 
# 

# In[15]:


def variable_expression_check(exp1: str, exp2: str, answer: dict[str: str]):
    if exp1 in exp2:
        return None
    else:
        if exp1 not in answer.keys():
            answer[exp1] = exp2
        return answer 
        


# In[16]:


unit_test = variable_expression_check("?x", "Cho`Gath", {})
assert unit_test == {'?x': 'Cho`Gath'}
unit_test = variable_expression_check("?x", "duplicate", {"?x" : "dont copy me"})
assert unit_test == {'?x': 'dont copy me'}
unit_test = variable_expression_check("?x", "?x", {"?x" : " this shouldnt show"})
assert unit_test == None


# <a id="check_keys"></a>
# ## check_keys
# 
# Due to the nature of the setup of this recursive algorithm, there is two dictionaries that will be combined at the end. For unification, if the same variable has multiple declarations, it should fail. This function checks the two results that the algorithm finds, and makes sure there is no duplicates in the function. If there is, the function returns false, and the algorithm fails.  **Used by**: [unification](#unification) 
# 
# * **result1**: dict[str: str] the dictionary that contains all the assignments for the expressions for the first part of the recursive algorithm.
# * **result2**: dict[str: str] the dictionary that contains all the assignments for the expressions for the second part of the recursive algorithm.
# 
# **returns**: bool, wil return false if there is no shared keys, true if there is any shared keys.

# In[17]:


def check_keys(result1, result2):
    for key in result1:
        if key in result2.keys():
            return True
    return False


# In[18]:


unit_test = check_keys({"?x" : "Cho`gath", "?y" : "Illaoi", "?z?" : "Sion"}, {"?a" : "Arthas", "?b" : "Frostmourne"})
assert unit_test == False
unit_test = check_keys({"?x" : "I am a duplicate"}, {"?x" : "me too"})
assert unit_test == True
unit_test = check_keys({}, {})
assert unit_test == False


# <a id="unification"></a>
# ## unification
# 
# This function uses the algorithm of unification to take in specific expression and assign variables to the expressions. It does not apply them to anything that would be read like a normal sentence, but sets up for that. So it will set up a substitution list that is assigned to variables to be used. It also has a series of checks to make sure that constants are not applied to different constants, and that variables have not been declared multiple times. If two constants are the same, the dictionary will have nothing added. The function will take in two list_expressions passed from unify, and if they have not been parse, parses them, or skips if they have been parsed, then moves on to the actual algorithm. The result that it returns is a dictionary of variables and their assignments, or None. **Uses**: [check_keys](#check_keys), [variable_expression_check](#variable_expression_check) **Used by**: [unify](#unify)
# 
# * **list_expression1**: list[str] or str: The first expression that is tested. Can be in either parsed or non-parsed format.
# * **list_expression2**: list[str] or str: The second expression that is tested. Can be in either parsed or non-parsed format.
# 
# **returns**: dict[str: str] or None: this function returns either the dictionary containing all the variables and their assignments, or None
# 

# In[19]:


def unification(list_expression1: str, list_expression2: str):
    if list_expression1 == [] and list_expression2 == []: return {}
    if list_expression1[0][0] == "(": exp1, exp2, answer = parse(list_expression1[0]), parse(list_expression2[0]), {}
    else: exp1, exp2, answer = list_expression1, list_expression2, {}
    if is_constant(exp1) and is_constant(exp2):
        if exp1 == exp2: return {}
        else: return None
    elif is_variable(exp1): return variable_expression_check(exp1, exp2, answer)
    elif is_variable(exp2): return variable_expression_check(exp2, exp1, answer)
    result1 = unification(exp1[0], exp2[0])
    if result1 is None: return None
    result2 = unification(exp1[1:], exp2[1:])
    if result2 is None: return None
    if check_keys(result1, result2): return None
    return result1 | result2


# In[20]:


def list_check(parsed_expression):
    if isinstance(parsed_expression, list):
        return parsed_expression
    return [parsed_expression]


# The `unification` pseudocode only takes lists so we have to make sure that we only pass a list.
# However, this has the side effect of making "foo" unify with ["foo"], at the start.
# That's ok.

# In[21]:


def unify( s_expression1, s_expression2):
    list_expression1 = list_check(s_expression1)
    list_expression2 = list_check(s_expression2)
    return unification( list_expression1, list_expression2)


# **Note** If you see the error,
# 
# ```
# tokenize.TokenError: ('EOF in multi-line statement', (2, 0))
# ```
# You most likely have unbalanced parentheses in your s-expression.
# 
# ## Test Cases
# 
# Use the expressions from the Self Check as your test cases...

# In[22]:


self_check_test_cases = [
    ['(son Barney Barney)', '(daughter Wilma Pebbles)', None],
    ['(Fred)', '(Barney)', None],
    ['(Pebbles)', '(Pebbles)', {}],
    ['(quarry_worker Fred)', '(quarry_worker ?x)', {'?x' : 'Fred'}],
    ['(son Barney ?x)', '(son ?y Bam_Bam)', {'?y' : 'Barney', '?x': 'Bam_Bam'}],
    ['(married ?x ?y)', '(married Barney Wilma)', {'?x' : 'Barney', '?y' : 'Wilma'}],
    ['(son Barney ?x)', '(son ?y (son Barney))', {'?y' :'Barney', '?x': ['son', 'Barney']}],
    ['(son Barney ?x)', '(son ?y (son ?y))', {'?y': 'Barney', '?x': ['son', '?y']}],
    ['(son Barney Bam_Bam)', '(son ?y (son Barney))', None],
    ['(loves Fred Fred)', '(loves ?x ?y)', {'?x' : 'Fred', '?y' :'Fred'}],
    ['(future George Fred)', '(future ?y ?y)', None]
]
for case in self_check_test_cases:
    exp1, exp2, expected = case
    actual = unify(exp1, exp2)
    print(f"actual = {actual}")
    print(f"expected = {expected}")
    print("\n")
    assert expected == actual


# Now add at least **five (5)** additional test cases of your own making, explaining exactly what you are testing. They should not be testing the same things as the self check test cases above.

# In[23]:


new_test_cases = [
    ['(son Barney Barney)', '(daughter Wilma Pebbles)', None, "non-equal constants"],
    ['(daughter ?x ?y)', '(daughter ?x ?y)', None, "Repeated variable declaration"],
    ['(friend ?x son ?y father ?z mother ?a)', '(friend Tagar son Baine_Bloodhoof father Cairne_Bloodhoof mother Tamaala_Bloodhoof)', {'?x' : 'Tagar', '?y': 'Baine_Bloodhoof', '?z': 'Cairne_Bloodhoof', '?a': 'Tamaala_Bloodhoof'}, "Testing long sets"],
    ['(chant ?x (FOR THE ALLAINCE))', '(chant (FOR THE HORDE LOK`TAR OGAR) ?y)', {'?x': ['FOR', 'THE', 'HORDE', 'LOK', '`', 'TAR', 'OGAR'], '?y': ['FOR', 'THE', 'ALLAINCE']}, "Testing extended parantheses"],
    ['( ?a ?b ?c ?d ?e)', '( ?f ?g ?h ?i ?j)', {'?a': '?f', '?b' : '?g', '?c' : '?h', '?d' : '?i', '?e' : '?j'}, "Testing only variables"],
    ['(allies ?Gondor ?Rohan)', '(allies (calls for aid) (and rohan will answer))', {'?Gondor' : ['calls', 'for', 'aid'], '?Rohan' : ['and', 'rohan', 'will', 'answer']}, "Testing long variable names and multiple parantheses in one statement"]
]
for case in new_test_cases:
    exp1, exp2, expected, message = case
    actual = unify(exp1, exp2)
    print(f"Testing {message}...")
    print(f"actual = {actual}")
    print(f"expected = {expected}")
    print("\n")
    assert expected == actual


# ## Before You Submit...
# 
# 1. Did you provide output exactly as requested?
# 2. Did you re-execute the entire notebook? ("Restart Kernel and Rull All Cells...")
# 3. If you did not complete the assignment or had difficulty please explain what gave you the most difficulty in the Markdown cell below.
# 4. Did you change the name of the file to `jhed_id.ipynb`?
# 
# Do not submit any other files.

# No specific comments on this one, just making sure I produced the output right. Something that I had to work around was that with the recursion of the expression, I had to set an if-else for the first time the expression was called, and if any other paranthesis were inserted into the expression. It worked in a manner that I assumed was acceptable, but wanted to make sure. Have a great week! 

# In[ ]:




