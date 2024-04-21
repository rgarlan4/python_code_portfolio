#!/usr/bin/env python
# coding: utf-8

# # Annotated follow-along guide: Loops and strings
# 
# This notebook contains the code used in the instructional videos from [Module 3: Loops and strings](https://www.coursera.org/learn/get-started-with-python/home/module/3).
# 
# ## Introduction
# 
# This follow-along guide is an annotated Jupyter Notebook organized to match the content from each module. It contains the same code shown in the videos for the module. In addition to content that is identical to what is covered in the videos, you’ll find additional information throughout the guide to explain the purpose of each concept covered, why the code is written in a certain way, and tips for running the code.
# 
# As you watch each of the following videos, an in-video message will appear to advise you that the video you are viewing contains coding instruction and examples. The in-video message will direct you to the relevant section in the notebook for the specific video you are viewing. Follow along in the notebook as the instructor discusses the code.
# 
# To skip directly to the code for a particular video, use the following links:
# 
# 1.   **[Introduction to while loops](#1)**
# 2.   **[Introduction to for loops](#2)**
# 3.   **[Loops with multiple range parameters](#3)**
# 4.   **[Work with strings](#4)**
# 5.   **[String slicing](#5)**
# 6.   **[Format strings](#6)**

# <a name="1"></a>
# ## 1. [Introduction to while loops](https://www.coursera.org/learn/get-started-with-python/lecture/M0dCL/introduction-to-while-loops) 

# In[1]:


# Instantiate a counter.
x = 0

# Create a while loop that prints "not there yet," increments x by 1, a 
# and prints x until x reaches 5.
while x < 5:
    print('Not there yet, x=' + str(x))
    x = x + 1
    print('x=' + str(x))


# In[ ]:


# Import the random module to be able to create a (pseudo) random number.
import random

number = random.randint(1,25)                   # Generate random number
number_of_guesses = 0                           # Instantiate guess counter

while number_of_guesses < 5:
    print('Guess a number between 1 and 25: ')  # Tell user to guess number
    guess = input()                             # Produce the user input field
    guess = int(guess)                          # Convert guess to integer
    number_of_guesses += 1                      # Increment guess count by 1

    if guess == number:                         # Break while loop if guess is correct
        break
    elif number_of_guesses == 5:                # Break while loop if guess limit reached
        break
    else:                                       # Tell user to try again
        print('Nope! Try again.')

# Message to display if correct
if guess == number:
    print('Correct! You guessed the number in ' + str(number_of_guesses) + ' tries!')
# Message to display after 5 unsuccessful guesses
else:
    print('You did not guess the number. The number was ' + str(number) + '.')


# <a name="2"></a>
# ## 2. [Introduction to for loops](https://www.coursera.org/learn/get-started-with-python/lecture/VKOIA/introduction-to-for-loops) 

# In[ ]:


# Example of for loop with range() function
for x in range(5):
    print(x)


# In[ ]:


# Example of reading in a .txt file line by line with a for loop
with open('zen_of_python.txt') as f:
    for line in f:
        print(line)
print('\nI\'m done.')


# <a name="3"></a>
# ## 3. [Loops with multiple range parameters](https://www.coursera.org/learn/get-started-with-python/lecture/2VI1Y/loops-with-multiple-range-parameters) 

# In[ ]:


# Use a for loop to calculate 9!
product = 1
for n in range(1, 10):
    product = product * n

print(product)


# In[ ]:


# Define a function that converts Fahrenheit to Celsius.
def to_celsius(x):
     return (x-32) * 5/9

# Create a table of Celsius-->Fahrenheit conversions every 10 degrees, 0-100
for x in range(0, 101, 10):
     print(x, to_celsius(x))


# <a name="4"></a>
# ## 4. [Work with strings](https://www.coursera.org/learn/get-started-with-python/lecture/k88nO/work-with-strings) 

# In[ ]:


# Adding strings will combine them.
'Hello' + 'world'


# In[ ]:


# Blank space ("whitespace") is its own character.
'Hello ' + 'world'


# In[ ]:


# Including a whitespace when combining strings
'Hello' + ' ' + 'world'


# In[ ]:


# Variables containing strings can be added.
greeting_1 = 'Hello '
greeting_2 = 'world'
greeting_1 + greeting_2


# In[ ]:


# Strings can be multiplied by integers.
danger = 'Danger! '
danger * 3


# In[ ]:


# Strings cannot be used with subtraction or division.
danger - 2


# In[ ]:


# Alternate single and double quotes to include one or the other in your string.
quote = '"Thank you for pressing the self-destruct button."'
print(quote)


# In[ ]:


# \ is an escape character that modifies the character that follows it.
quote = "\"It's dangerous to go alone!\""
print(quote)


# In[ ]:


# \n creates a newline.
greeting = "Good day,\nsir."
print(greeting)


# In[ ]:


# Using escape character (\) lets you express the newline symbol within a string.
newline = "\\n represents a newline in Python."
print(newline)


# In[ ]:


# You can loop over strings.
python = 'Python'
for letter in python:
    print(letter + 'ut')


# <a name="5"></a>
# ## 5. [String slicing](https://www.coursera.org/learn/get-started-with-python/lecture/7741K/string-slicing) 

# In[ ]:


# The index() method returns index of character's first occurrence in string.
pets = 'cats and dogs'
pets.index('s')


# In[ ]:


# The index() method will throw an error if character is not in string.
pets.index('z')


# In[ ]:


# Access the character at a given index of a string.
name = 'Jolene'
name[0]


# In[ ]:


# Access the character at a given index of a string.
name[5]


# In[ ]:


# Indices that are out of range will return an IndexError.
name[6]


# In[ ]:


# Negative indexing begins at the end of the string.
sentence = 'A man, a plan, a canal, Panama!'
sentence[-1]


# In[ ]:


# Negative indexing begins at the end of the string.
sentence[-2]


# In[ ]:


# Access a substring by using a slice.
color = 'orange'
color[1:4]


# In[ ]:


# Omitting the first value of the slice implies a value of 0.
fruit = 'pineapple'
fruit[:4]


# In[ ]:


# Omitting the last value of the slice implies a value of len(string).
fruit[4:]


# In[ ]:


# The `in` keyword returns Boolean of whether substring is in string.
'banana' in fruit


# In[ ]:


# The `in` keyword returns Boolean of whether substring is in string.
'apple' in fruit


# <a name="6"></a>
# ## 6. [Format strings](https://www.coursera.org/learn/get-started-with-python/lecture/mYMRp/format-strings) 

# In[ ]:


# Use format() method to insert values into your string, indicated by braces.
name = 'Manuel'
number = 3
print('Hello {}, your lucky number is {}.'.format(name, number))


# In[ ]:


# You can assign names to designate how you want values to be inserted.
name = 'Manuel'
number = 3
print('Hello {name}, your lucky number is {num}.'.format(num=number, name=name))


# In[ ]:


# You can use argument indices to designate how you want values to be inserted.
print('Hello {1}, your lucky number is {0}.'.format(number, name))


# In[ ]:


# Example inserting prices into string
price = 7.75
with_tax = price * 1.07
print('Base price: ${} USD. \nWith tax: ${} USD.'.format(price, with_tax))


# In[ ]:


# Use :.2f to round a float value to two places beyond the decimal.
print('Base price: ${:.2f} USD. \nWith tax: ${:.2f} USD.'.format(price, with_tax))


# In[ ]:


# Define a function that converts Fahrenheit to Celsius.
def to_celsius(x):
    return (x-32) * 5/9

# Create a temperature conversion table using string formatting
for x in range(0, 101, 10):
    print("{:>3} F | {:>6.2f} C".format(x, to_celsius(x)))


# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

# In[ ]:





# In[ ]:





# In[ ]:




