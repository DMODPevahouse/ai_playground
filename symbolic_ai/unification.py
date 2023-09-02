import tokenize
from io import StringIO


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


def is_variable( exp):
    return isinstance( exp, str) and exp[ 0] == "?"


def is_constant( exp):
    return isinstance( exp, str) and not is_variable( exp)


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
unit_test = variable_expression_check("?x", "duplicate", {"?x": "dont copy me"})
assert unit_test == {'?x': 'dont copy me'}
unit_test = variable_expression_check("?x", "?x", {"?x": " this shouldnt show"})
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


unit_test = check_keys({"?x": "Cho`gath", "?y": "Illaoi", "?z?": "Sion"}, {"?a": "Arthas", "?b": "Frostmourne"})
assert unit_test == False
unit_test = check_keys({"?x": "I am a duplicate"}, {"?x": "me too"})
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
    if list_expression1[0][0] == "(":
        exp1, exp2, answer = parse(list_expression1[0]), parse(list_expression2[0]), {}
    else:
        exp1, exp2, answer = list_expression1, list_expression2, {}
    if is_constant(exp1) and is_constant(exp2):
        if exp1 == exp2:
            return {}
        else:
            return None
    elif is_variable(exp1):
        return variable_expression_check(exp1, exp2, answer)
    elif is_variable(exp2):
        return variable_expression_check(exp2, exp1, answer)
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




# In[21]:


def unify(s_expression1, s_expression2):
    list_expression1 = list_check(s_expression1)
    list_expression2 = list_check(s_expression2)
    return unification(list_expression1, list_expression2)



# In[22]:


self_check_test_cases = [
    ['(son Barney Barney)', '(daughter Wilma Pebbles)', None],
    ['(Fred)', '(Barney)', None],
    ['(Pebbles)', '(Pebbles)', {}],
    ['(quarry_worker Fred)', '(quarry_worker ?x)', {'?x': 'Fred'}],
    ['(son Barney ?x)', '(son ?y Bam_Bam)', {'?y': 'Barney', '?x': 'Bam_Bam'}],
    ['(married ?x ?y)', '(married Barney Wilma)', {'?x': 'Barney', '?y': 'Wilma'}],
    ['(son Barney ?x)', '(son ?y (son Barney))', {'?y': 'Barney', '?x': ['son', 'Barney']}],
    ['(son Barney ?x)', '(son ?y (son ?y))', {'?y': 'Barney', '?x': ['son', '?y']}],
    ['(son Barney Bam_Bam)', '(son ?y (son Barney))', None],
    ['(loves Fred Fred)', '(loves ?x ?y)', {'?x': 'Fred', '?y': 'Fred'}],
    ['(future George Fred)', '(future ?y ?y)', None]
]
for case in self_check_test_cases:
    exp1, exp2, expected = case
    actual = unify(exp1, exp2)
    print(f"actual = {actual}")
    print(f"expected = {expected}")
    print("\n")
    assert expected == actual


# In[23]:


new_test_cases = [
    ['(son Barney Barney)', '(daughter Wilma Pebbles)', None, "non-equal constants"],
    ['(daughter ?x ?y)', '(daughter ?x ?y)', None, "Repeated variable declaration"],
    ['(friend ?x son ?y father ?z mother ?a)',
     '(friend Tagar son Baine_Bloodhoof father Cairne_Bloodhoof mother Tamaala_Bloodhoof)',
     {'?x': 'Tagar', '?y': 'Baine_Bloodhoof', '?z': 'Cairne_Bloodhoof', '?a': 'Tamaala_Bloodhoof'},
     "Testing long sets"],
    ['(chant ?x (FOR THE ALLAINCE))', '(chant (FOR THE HORDE LOK`TAR OGAR) ?y)',
     {'?x': ['FOR', 'THE', 'HORDE', 'LOK', '`', 'TAR', 'OGAR'], '?y': ['FOR', 'THE', 'ALLAINCE']},
     "Testing extended parantheses"],
    ['( ?a ?b ?c ?d ?e)', '( ?f ?g ?h ?i ?j)', {'?a': '?f', '?b': '?g', '?c': '?h', '?d': '?i', '?e': '?j'},
     "Testing only variables"],
    ['(allies ?Gondor ?Rohan)', '(allies (calls for aid) (and rohan will answer))',
     {'?Gondor': ['calls', 'for', 'aid'], '?Rohan': ['and', 'rohan', 'will', 'answer']},
     "Testing long variable names and multiple parantheses in one statement"]
]
for case in new_test_cases:
    exp1, exp2, expected, message = case
    actual = unify(exp1, exp2)
    print(f"Testing {message}...")
    print(f"actual = {actual}")
    print(f"expected = {expected}")
    print("\n")
    assert expected == actual
