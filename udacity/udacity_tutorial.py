# Udacity
# Variables
x, y, z = 3, 4, 5
y

# data types: integer, float, string , bool
int(49.7)
print(.1 + .1 + .1 == .3)


salesman = 'I think you\'re an "encyclopedia" salesman'
print("test"*8)
print(len("ababa") / len("ab"))

# arithmetic operators  + - / * // ** (symbols)
# assignment operators = +=
# comparison operators < > == !=
# logical operators and or not
# identity operators is is not

# functions print() len() float()

# Process data / operating on values with operators (short symbols) and functions (descriptive names) and methods (associated with specific types of objects (the data type for a particular variable) / Methods actually are functions that are called using dot notation)
# String Methods
salesman.title()
salesman.lower()
salesman.islower()
salesman.count("a", 20, 40)
str(dir(str)).upper()
https://docs.python.org/3/library/stdtypes.html#string-methods

animal = "dog"
action = "bite"
print("Does your {} {}?".format(animal, action))

str4 = ("dog", "bite")
print("Does your %s %s ?" % str4)

maria_string = "Maria loves {} and {}"
print(maria_string.format("math", "statistics"))

new_str = "The cow jumped over the moon."
new_str.split(None, 3)

new_str.find("moon")
new_str.rfind("moon")

# Types of Data Structures (data containers): Lists, Tuples, Sets, Dictionaries, 
# Compound Data Structures
# Operators: Membership, Identity

#List is a data structure that is mutable ordered sequence of elements
# List Methods

list_of_random_things = [1, 3.4, 'a string', True]
list_of_random_things[1:2]
list_of_random_things[1:]


'this' in 'this is a string'
5 not in [1, 2, 3, 4, 6]
5 in [1, 2, 3, 4, 6]
[1, 2, 3, 4, 6][-3:]
list_of_random_things[-2:] = [99, 98]

VINIX = ['C', 'MA', 'BA', 'PG', 'CSCO', 'VZ', 'PFE', 'HD', 'INTC', 'T', 'V', 'UNH', 'WFC', 'CVX', 'BAC', 'JNJ', 'GOOGL', 'GOOG', 'BRK.B', 'XOM', 'JPM', 'FB', 'AMZN', 'MSFT', 'AAPL']
"GOOGL" in VINIX
sorted(VINIX, reverse=True)


letters = ['a', 'b', 'c', 'd']
let2 = letters
letters, let2
letters[0] = "x"
letters, let2
let2[0] = "y"
letters, let2

new_str = "\n".join(["fore", "aft", "starboard", "port"])
print(new_str)

letters = ['a', 'b', 'c', 'd']
letters.append('z')
print(letters)



# Tuple
# It's a data type for immutable ordered sequences of elements.
# A tuple is an immutable, ordered data structure that can be indexed and sliced like a list.
#  Tuples are defined by listing a sequence of elements separated by commas, optionally contained within parentheses: ().

dimensions = 52, 40, 100
length, width, height = dimensions
print("The dimensions are {} x {} x {}".format(length, width, height))



# Set 
# A set is a data type for mutable unordered collections of unique elements.

numbers = [1, 2, 6, 3, 1, 1, 6]
unique_nums = set(numbers)
print(unique_nums)

fruit = {"apple", "banana", "orange", "grapefruit"}  # define a set
print("watermelon" in fruit)  # check for element
fruit.add("watermelon")  # add an element
print(fruit)
print(fruit.pop())  # remove a random element
print(fruit)



# Dictionaries
#  A dictionary is a mutable data type that stores mappings of unique keys to values. 

elements = {"hydrogen": 1, "helium": 2, "carbon": 6}
elements["helium"]
elements["lithium"] = 3
"carbon" in elements
elements.get("carbon")
print(elements.get("dilithium"))
n = elements.get("dilithium")
print(n is None)
print(n is not None)
elements.get('kryptonite', 'There\'s no such element!')


# In Python, any immutable object (such as an integer, boolean, string, tuple) is hashable, meaning its value 
# does not change during its lifetime. 

elements = {
    "hydrogen": {
        "number": 1,
        "weight": 1.00794,
        "symbol": "H"
    },
    "helium": {
        "number": 2,
        "weight": 4.002602,
        "symbol": "He"
    }
}

elements["helium"]
elements["helium"]["weight"]

oxygen = {"number": 8, "weight": 15.999, "symbol": "O"}
elements["oxygen"] = oxygen
elements

elements["hydrogen"]["is_noble_gas"] = False
elements["helium"]["is_noble_gas"] = True
elements

sorted(elements.keys())
"hydrogen" in elements
elements.get("hydrogen")


# Quiz: Verse Dictionary
verse_dict =  {'if': 3, 'you': 6, 'can': 3, 'keep': 1, 'your': 1, 'head': 1, 'when': 2, 'all': 2, 'about': 2, 'are': 1, 'losing': 1, 'theirs': 1, 'and': 3, 'blaming': 1, 'it': 1, 'on': 1, 'trust': 1, 'yourself': 1, 'men': 1, 'doubt': 1, 'but': 1, 'make': 1, 'allowance': 1, 'for': 1, 'their': 1, 'doubting': 1, 'too': 3, 'wait': 1, 'not': 1, 'be': 1, 'tired': 1, 'by': 1, 'waiting': 1, 'or': 2, 'being': 2, 'lied': 1, 'don\'t': 3, 'deal': 1, 'in': 1, 'lies': 1, 'hated': 1, 'give': 1, 'way': 1, 'to': 1, 'hating': 1, 'yet': 1, 'look': 1, 'good': 1, 'nor': 1, 'talk': 1, 'wise': 1}
print(verse_dict, '\n')

# find number of unique keys in the dictionary
num_keys = len(verse_dict)
print(num_keys)

# find whether 'breathe' is a key in the dictionary
contains_breathe = verse_dict.get("breathe")
print(contains_breathe)
print("breathe" in verse_dict)

# create and sort a list of the dictionary's keys
sorted_keys = sorted(verse_dict.keys())

# get the first element in the sorted list of keys
print(sorted_keys[0])

# find the element with the highest value in the list of keys
print(sorted_keys[-1])
# find the element with the highest value in the list of values
import operator
print(max(verse_dict.items(), key=operator.itemgetter(1))) 


Data Structure	   Ordered	Mutable	Constructor	   Example
List	            Yes	   Yes	   [ ] or list()	[5.7, 4, 'yes', 5.7]
Tuple	            Yes	   No	      ( ) or tuple()	(5.7, 4, 'yes', 5.7)
Set	            No	      Yes	   {}* or set()	{5.7, 4, 'yes'}
Dictionary	      No	      No**	   { } or dict()	{'Jun': 75, 'Jul': 89}



# Control flow
 # If, Elif, Else Statements

season = "autumn"

if season == 'spring':
   print('plant the garden!')
elif season == 'summer':
   print('water the garden!')
elif season == 'fall':
   print('harvest the garden!')
elif season == 'winter':
   print('stay indoors!')
else:
   print('unrecognized season')


is_cold = True
if is_cold == True:
   print("The weather is cold!")
if is_cold == True:
   print("The weather is cold!")   

# Here are most of the built-in objects that are considered False in Python:
# constants defined to be false: None and False
# zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
# empty sequences and collections: '"", (), [], {}, set(), range(0)



# For
cities = ['new york city', 'mountain view', 'chicago', 'los angeles']
for city in cities:
   print(city)
print("Done!")


cities = ['new york city', 'mountain view', 'chicago', 'los angeles']
for index in range(len(cities)):
   print(cities[index])
   # cities[index] = cities[index].title()


names = ["Joey Tribbiani", "Monica Geller", "Chandler Bing", "Phoebe Buffay"]
usernames = []
# write your for loop here
for name in names:
   usernames.append(names[i].lower().replace(" ", "_"))
print(usernames)


items = ['first string', 'second string']
html_str = "<ul>\n"  # "\ n" is the character that marks the end of the line, it does
                     # the characters that are after it in html_str are on the next line
# write your code here
for item in items:
   html_str += "<li>{}</li>\n".format(item)
html_str += "</ul>"
print(html_str)



# Building Dictionaries

book_title =  ['great', 'expectations','the', 'adventures', 'of', 'sherlock','holmes','the','great','gasby','hamlet','adventures','of','huckleberry','fin']
word_counter = {}
for word in book_title:
   if word not in word_counter:
      word_counter[word] = 1
   else:
      word_counter[word] += 1
word_counter


book_title =  ['great', 'expectations','the', 'adventures', 'of', 'sherlock','holmes','the','great','gasby','hamlet','adventures','of','huckleberry','fin']
word_counter = {}
for word in book_title:
   word_counter[word] = word_counter.get(word, 0) + 1
word_counter


cast = {
   "Jerry Seinfeld": "Jerry Seinfeld",
   "Julia Louis-Dreyfus": "Elaine Benes",
   "Jason Alexander": "George Costanza",
   "Michael Richards": "Cosmo Kramer"
}
for key, value in cast.items():
   print("actor: {}, character: {}".format(key, value))


result = 0
basket_items = {'apples': 4, 'oranges': 19, 'kites': 3, 'sandwiches': 8}
fruits = ['apples', 'oranges', 'pears', 'peaches', 'grapes', 'bananas']
#Iterate through the dictionary if the key is in the list of fruits, add the value (number of fruits) to result
for item in basket_items:
   if item in fruits:
      result += basket_items[item]
print(result)

result = 0
basket_items = {'apples': 4, 'oranges': 19, 'kites': 3, 'sandwiches': 8}
fruits = ['apples', 'oranges', 'pears', 'peaches', 'grapes', 'bananas']
for key, val in basket_items.items():
   if key in fruits:
      result += val
print(result)



# While

card_deck = [4, 11, 8, 5, 13, 2, 8, 10]
hand = []

# adds the last element of the card_deck list to the hand list
# until the values in hand add up to 17 or more
while sum(hand) < 17:
   hand.append(card_deck.pop())
print(hand)



num_list = [422, 136, 524, 85, 96, 719, 85, 92, 10, 17, 312, 542, 87, 23, 86, 191, 116, 35, 173, 45, 149, 59, 84, 69, 113, 166]
odd, su = 0, 0
for num in num_list:
   if odd < 5 and num % 2 != 0:
      su += num
      odd += 1
print(su)

odd, su, i = 0, 0, 0
while odd < 5 and i < len(num_list):
   if num_list[i] % 2 != 0:
      su += num_list[i]
      odd += 1
   i += 1   
print(su)



# Break, Continue

# break terminates a loop
# continue skips one iteration of a loop

manifest = [("bananas", 15), ("mattresses", 24), ("dog kennels", 42), ("machine", 120), ("cheeses", 5)]
weight = 0
items = []
for cargo_name, cargo_weight in manifest:
   print("current weight: {}".format(weight))
   if weight >= 100:
      print("  breaking from the loop now!")
      break
   elif weight + cargo_weight > 100:
      print("  skipping {} ({})".format(cargo_name, cargo_weight))
      continue
   else:
      print("  adding {} ({})".format(cargo_name, cargo_weight))
      items.append(cargo_name)
      weight += cargo_weight

print("\nFinal Weight: {}".format(weight))
print("Final Items: {}".format(items))



# Write a loop with a break statement to create a string, news_ticker, that is exactly 140 characters long. You should create the news ticker by adding headlines from the headlines list, inserting a space in between each headline. If necessary, truncate the last headline in the middle so that news_ticker is exactly 140 characters long.
headlines = ["Local Bear Eaten by Man",
             "Legislature Announces New Laws",
             "Peasant Discovers Violence Inherent in System",
             "Cat Rescues Fireman Stuck in Tree",
             "Brave Knight Runs Away",
             "Papperbok Review: Totally Triffic"]
news_ticker = ""

for head in headlines:
   if len(news_ticker) > 140:
      break
   elif len(news_ticker) + len(head) > 140:
      tail_ind = len(head) - ((len(news_ticker) + len(head)) - 140)
      news_ticker += head[:tail_ind]
      break
   else:
      news_ticker += ("{} ".format(head)) # news_ticker += head + " "
print(news_ticker)


headlines = ["Local Bear Eaten by Man",
             "Legislature Announces New Laws",
             "Peasant Discovers Violence Inherent in System",
             "Cat Rescues Fireman Stuck in Tree",
             "Brave Knight Runs Away",
             "Papperbok Review: Totally Triffic"]

news_ticker = ""
for headline in headlines:
   news_ticker += headline + " "
   if len(news_ticker) >= 140:
      news_ticker = news_ticker[:140]
      break
print(news_ticker)



## Your code should check if each number in the list is a prime number
check_prime = [26, 39, 51, 53, 57, 79, 85]
for check_pri in check_prime:
   odd_q = True
   for i in range(2, check_pri//2):
      if check_pri % i == 0:
         odd_q = False
         break
   if odd_q:
      print("{} is a prime number".format(check_pri))
   else:
      print("{} is not a prime number, because {} is a factor".format(check_pri, i))


check_prime = [26, 39, 51, 53, 57, 79, 85]
# iterate through the check_prime list
for num in check_prime:
# search for factors, iterating through numbers ranging from 2 to the number itself
   for i in range(2, num):
# number is not prime if modulo is 0
      if (num % i) == 0:
         print("{} is NOT a prime number, because {} is a factor of {}".format(num, i, num))
         break
# otherwise keep checking until we've searched all possible factors, and then declare it prime
      if i == num//2 -1:    
         print("{} IS a prime number".format(num))



# Zip & Enumerate

letters1 = ['a', 'b', 'c']
nums1 = [1, 2, 3]
list(zip(letters1, nums1))
for letter, num in zip(letters1, nums1):
   print("{}: {}".format(letter, num))

letters2, nums2 = zip(*zip(letters1, nums1))
letters2, nums2

letters1 = ['a', 'b', 'c', 'd', 'e']
list(enumerate(letters1))
for i, letter in enumerate(letters1):
   print(i, letter)


x_coord = [23, 53, 2, -12, 95, 103, 14, -5]
y_coord = [677, 233, 405, 433, 905, 376, 432, 445]
z_coord = [4, 16, -6, -42, 3, -6, 23, -1]
labels = ["F", "J", "A", "Q", "Y", "B", "W", "X"]
points = []
for i in zip(labels, x_coord, y_coord, z_coord):
   points.append("{}: {}, {}, {}".format(*i))
for point in points:
   print(point)


cast_names = ["Barney", "Robin", "Ted", "Lily", "Marshall"]
cast_heights = [72, 68, 72, 66, 76]
cast = dict(zip(cast_names, cast_heights))
print(cast)


cast = (("Barney", 72), ("Robin", 68), ("Ted", 72), ("Lily", 66), ("Marshall", 76))
names, heights = zip(*cast)
print(names)
print(heights)


data = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
data_transpose = tuple(zip(*data)) # replace with your code
print(data_transpose)


cast = ["Barney Stinson", "Robin Scherbatsky", "Ted Mosby", "Lily Aldrin", "Marshall Eriksen"]
heights = [72, 68, 72, 66, 76]
# write your for loop here
for i, j in enumerate(cast):
   # cast[i] = "{} {}".format(j, i)
   cast[i] = j +  " " + str(heights[i])
print(cast)


# List Comprehensions

squares = [x**2 for x in range(9)]
squares = [x**2 for x in range(9) if x % 2 == 0]
squares = [x**2 if x % 2 == 0 else x + 3 for x in range(9)]

names = ["Rick Sanchez", "Morty Smith", "Summer Smith", "Jerry Smith", "Beth Smith"]
first_names = [name.split()[0].lower() for name in names]
print(first_names)


scores = {
            "Rick Sanchez": 70,
            "Morty Smith": 35,
            "Summer Smith": 82,
            "Jerry Smith": 23,
            "Beth Smith": 98
         }
passed = [key for key, val in scores.items() if val >= 65]
print(passed)



nominated = {1931: ['Norman Taurog', 'Wesley Ruggles', 'Clarence Brown', 'Lewis Milestone', 'Josef Von Sternberg'], 1932: ['Frank Borzage', 'King Vidor', 'Josef Von Sternberg'], 1933: ['Frank Lloyd', 'Frank Capra', 'George Cukor'], 1934: ['Frank Capra', 'Victor Schertzinger', 'W. S. Van Dyke'], 1935: ['John Ford', 'Michael Curtiz', 'Henry Hathaway', 'Frank Lloyd'], 1936: ['Frank Capra', 'William Wyler', 'Robert Z. Leonard', 'Gregory La Cava', 'W. S. Van Dyke'], 1937: ['Leo McCarey', 'Sidney Franklin', 'William Dieterle', 'Gregory La Cava', 'William Wellman'], 1938: ['Frank Capra', 'Michael Curtiz', 'Norman Taurog', 'King Vidor', 'Michael Curtiz'], 1939: ['Sam Wood', 'Frank Capra', 'John Ford', 'William Wyler', 'Victor Fleming'], 1940: ['John Ford', 'Sam Wood', 'William Wyler', 'George Cukor', 'Alfred Hitchcock'], 1941: ['John Ford', 'Orson Welles', 'Alexander Hall', 'William Wyler', 'Howard Hawks'], 1942: ['Sam Wood', 'Mervyn LeRoy', 'John Farrow', 'Michael Curtiz', 'William Wyler'], 1943: ['Michael Curtiz', 'Ernst Lubitsch', 'Clarence Brown', 'George Stevens', 'Henry King'], 1944: ['Leo McCarey', 'Billy Wilder', 'Otto Preminger', 'Alfred Hitchcock', 'Henry King'], 1945: ['Billy Wilder', 'Leo McCarey', 'Clarence Brown', 'Jean Renoir', 'Alfred Hitchcock'], 1946: ['David Lean', 'Frank Capra', 'Robert Siodmak', 'Clarence Brown', 'William Wyler'], 1947: ['Elia Kazan', 'Henry Koster', 'Edward Dmytryk', 'George Cukor', 'David Lean'], 1948: ['John Huston', 'Laurence Olivier', 'Jean Negulesco', 'Fred Zinnemann', 'Anatole Litvak'], 1949: ['Joseph L. Mankiewicz', 'Robert Rossen', 'William A. Wellman', 'Carol Reed', 'William Wyler'], 1950: ['Joseph L. Mankiewicz', 'John Huston', 'George Cukor', 'Billy Wilder', 'Carol Reed'], 1951: ['George Stevens', 'John Huston', 'Vincente Minnelli', 'William Wyler', 'Elia Kazan'], 1952: ['John Ford', 'Joseph L. Mankiewicz', 'Cecil B. DeMille', 'Fred Zinnemann', 'John Huston'], 1953: ['Fred Zinnemann', 'Charles Walters', 'William Wyler', 'George Stevens', 'Billy Wilder'], 1954: ['Elia Kazan', 'George Seaton', 'William Wellman', 'Alfred Hitchcock', 'Billy Wilder'], 1955: ['Delbert Mann', 'John Sturges', 'Elia Kazan', 'Joshua Logan', 'David Lean'], 1956: ['George Stevens', 'Michael Anderson', 'William Wyler', 'Walter Lang', 'King Vidor'], 1957: ['David Lean', 'Mark Robson', 'Joshua Logan', 'Sidney Lumet', 'Billy Wilder'], 1958: ['Richard Brooks', 'Stanley Kramer', 'Robert Wise', 'Mark Robson', 'Vincente Minnelli'], 1959: ['George Stevens', 'Fred Zinnemann', 'Jack Clayton', 'Billy Wilder', 'William Wyler'], 1960: ['Billy Wilder', 'Jules Dassin', 'Alfred Hitchcock', 'Jack Cardiff', 'Fred Zinnemann'], 1961: ['J. Lee Thompson', 'Robert Rossen', 'Stanley Kramer', 'Federico Fellini', 'Robert Wise', 'Jerome Robbins'], 1962: ['David Lean', 'Frank Perry', 'Pietro Germi', 'Arthur Penn', 'Robert Mulligan'], 1963: ['Elia Kazan', 'Otto Preminger', 'Federico Fellini', 'Martin Ritt', 'Tony Richardson'], 1964: ['George Cukor', 'Peter Glenville', 'Stanley Kubrick', 'Robert Stevenson', 'Michael Cacoyannis'], 1965: ['William Wyler', 'John Schlesinger', 'David Lean', 'Hiroshi Teshigahara', 'Robert Wise'], 1966: ['Fred Zinnemann', 'Michelangelo Antonioni', 'Claude Lelouch', 'Richard Brooks', 'Mike Nichols'], 1967: ['Arthur Penn', 'Stanley Kramer', 'Richard Brooks', 'Norman Jewison', 'Mike Nichols'], 1968: ['Carol Reed', 'Gillo Pontecorvo', 'Anthony Harvey', 'Franco Zeffirelli', 'Stanley Kubrick'], 1969: ['John Schlesinger', 'Arthur Penn', 'George Roy Hill', 'Sydney Pollack', 'Costa-Gavras'], 1970: ['Franklin J. Schaffner', 'Federico Fellini', 'Arthur Hiller', 'Robert Altman', 'Ken Russell'], 1971: ['Stanley Kubrick', 'Norman Jewison', 'Peter Bogdanovich', 'John Schlesinger', 'William Friedkin'], 1972: ['Bob Fosse', 'John Boorman', 'Jan Troell', 'Francis Ford Coppola', 'Joseph L. Mankiewicz'], 1973: ['George Roy Hill', 'George Lucas', 'Ingmar Bergman', 'William Friedkin', 'Bernardo Bertolucci'], 1974: ['Francis Ford Coppola', 'Roman Polanski', 'Francois Truffaut', 'Bob Fosse', 'John Cassavetes'], 1975: ['Federico Fellini', 'Stanley Kubrick', 'Sidney Lumet', 'Robert Altman', 'Milos Forman'], 1976: ['Alan J. Pakula', 'Ingmar Bergman', 'Sidney Lumet', 'Lina Wertmuller', 'John G. Avildsen'], 1977: ['Steven Spielberg', 'Fred Zinnemann', 'George Lucas', 'Herbert Ross', 'Woody Allen'], 1978: ['Hal Ashby', 'Warren Beatty', 'Buck Henry', 'Woody Allen', 'Alan Parker', 'Michael Cimino'], 1979: ['Bob Fosse', 'Francis Coppola', 'Peter Yates', 'Edouard Molinaro', 'Robert Benton'], 1980: ['David Lynch', 'Martin Scorsese', 'Richard Rush', 'Roman Polanski', 'Robert Redford'], 1981: ['Louis Malle', 'Hugh Hudson', 'Mark Rydell', 'Steven Spielberg', 'Warren Beatty'], 1982: ['Wolfgang Petersen', 'Steven Spielberg', 'Sydney Pollack', 'Sidney Lumet', 'Richard Attenborough'], 1983: ['Peter Yates', 'Ingmar Bergman', 'Mike Nichols', 'Bruce Beresford', 'James L. Brooks'], 1984: ['Woody Allen', 'Roland Joffe', 'David Lean', 'Robert Benton', 'Milos Forman'], 1985: ['Hector Babenco', 'John Huston', 'Akira Kurosawa', 'Peter Weir', 'Sydney Pollack'], 1986: ['David Lynch', 'Woody Allen', 'Roland Joffe', 'James Ivory', 'Oliver Stone'], 1987: ['Bernardo Bertolucci', 'Adrian Lyne', 'John Boorman', 'Norman Jewison', 'Lasse Hallstrom'], 1988: ['Barry Levinson', 'Charles Crichton', 'Martin Scorsese', 'Alan Parker', 'Mike Nichols'], 1989: ['Woody Allen', 'Peter Weir', 'Kenneth Branagh', 'Jim Sheridan', 'Oliver Stone'], 1990: ['Francis Ford Coppola', 'Martin Scorsese', 'Stephen Frears', 'Barbet Schroeder', 'Kevin Costner'], 1991: ['John Singleton', 'Barry Levinson', 'Oliver Stone', 'Ridley Scott', 'Jonathan Demme'], 1992: ['Clint Eastwood', 'Neil Jordan', 'James Ivory', 'Robert Altman', 'Martin Brest'], 1993: ['Jim Sheridan', 'Jane Campion', 'James Ivory', 'Robert Altman', 'Steven Spielberg'], 1994: ['Woody Allen', 'Quentin Tarantino', 'Robert Redford', 'Krzysztof Kieslowski', 'Robert Zemeckis'], 1995: ['Chris Noonan', 'Tim Robbins', 'Mike Figgis', 'Michael Radford', 'Mel Gibson'], 1996: ['Anthony Minghella', 'Joel Coen', 'Milos Forman', 'Mike Leigh', 'Scott Hicks'], 1997: ['Peter Cattaneo', 'Gus Van Sant', 'Curtis Hanson', 'Atom Egoyan', 'James Cameron'], 1998: ['Roberto Benigni', 'John Madden', 'Terrence Malick', 'Peter Weir', 'Steven Spielberg'], 1999: ['Spike Jonze', 'Lasse Hallstrom', 'Michael Mann', 'M. Night Shyamalan', 'Sam Mendes'], 2000: ['Stephen Daldry', 'Ang Lee', 'Steven Soderbergh', 'Ridley Scott', 'Steven Soderbergh'], 2001: ['Ridley Scott', 'Robert Altman', 'Peter Jackson', 'David Lynch', 'Ron Howard'], 2002: ['Rob Marshall', 'Martin Scorsese', 'Stephen Daldry', 'Pedro Almodovar', 'Roman Polanski'], 2003: ['Fernando Meirelles', 'Sofia Coppola', 'Peter Weir', 'Clint Eastwood', 'Peter Jackson'], 2004: ['Martin Scorsese', 'Taylor Hackford', 'Alexander Payne', 'Mike Leigh', 'Clint Eastwood'], 2005: ['Ang Lee', 'Bennett Miller', 'Paul Haggis', 'George Clooney', 'Steven Spielberg'], 2006: ['Alejandro Gonzaalez Inarritu', 'Clint Eastwood', 'Stephen Frears', 'Paul Greengrass', 'Martin Scorsese'], 2007: ['Julian Schnabel', 'Jason Reitman', 'Tony Gilroy', 'Paul Thomas Anderson', 'Joel Coen', 'Ethan Coen'], 2008: ['David Fincher', 'Ron Howard', 'Gus Van Sant', 'Stephen Daldry', 'Danny Boyle'], 2009: ['James Cameron', 'Quentin Tarantino', 'Lee Daniels', 'Jason Reitman', 'Kathryn Bigelow'], 2010: ['Darren Aronofsky', 'David O. Russell', 'David Fincher', 'Ethan Coen', 'Joel Coen', 'Tom Hooper']}
winners = {1931: ['Norman Taurog'], 1932: ['Frank Borzage'], 1933: ['Frank Lloyd'], 1934: ['Frank Capra'], 1935: ['John Ford'], 1936: ['Frank Capra'], 1937: ['Leo McCarey'], 1938: ['Frank Capra'], 1939: ['Victor Fleming'], 1940: ['John Ford'], 1941: ['John Ford'], 1942: ['William Wyler'], 1943: ['Michael Curtiz'], 1944: ['Leo McCarey'], 1945: ['Billy Wilder'], 1946: ['William Wyler'], 1947: ['Elia Kazan'], 1948: ['John Huston'], 1949: ['Joseph L. Mankiewicz'], 1950: ['Joseph L. Mankiewicz'], 1951: ['George Stevens'], 1952: ['John Ford'], 1953: ['Fred Zinnemann'], 1954: ['Elia Kazan'], 1955: ['Delbert Mann'], 1956: ['George Stevens'], 1957: ['David Lean'], 1958: ['Vincente Minnelli'], 1959: ['William Wyler'], 1960: ['Billy Wilder'], 1961: ['Jerome Robbins', 'Robert Wise'], 1962: ['David Lean'], 1963: ['Tony Richardson'], 1964: ['George Cukor'], 1965: ['Robert Wise'], 1966: ['Fred Zinnemann'], 1967: ['Mike Nichols'], 1968: ['Carol Reed'], 1969: ['John Schlesinger'], 1970: ['Franklin J. Schaffner'], 1971: ['William Friedkin'], 1972: ['Bob Fosse'], 1973: ['George Roy Hill'], 1974: ['Francis Ford Coppola'], 1975: ['Milos Forman'], 1976: ['John G. Avildsen'], 1977: ['Woody Allen'], 1978: ['Michael Cimino'], 1979: ['Robert Benton'], 1980: ['Robert Redford'], 1981: ['Warren Beatty'], 1982: ['Richard Attenborough'], 1983: ['James L. Brooks'], 1984: ['Milos Forman'], 1985: ['Sydney Pollack'], 1986: ['Oliver Stone'], 1987: ['Bernardo Bertolucci'], 1988: ['Barry Levinson'], 1989: ['Oliver Stone'], 1990: ['Kevin Costner'], 1991: ['Jonathan Demme'], 1992: ['Clint Eastwood'], 1993: ['Steven Spielberg'], 1994: ['Robert Zemeckis'], 1995: ['Mel Gibson'], 1996: ['Anthony Minghella'], 1997: ['James Cameron'], 1998: ['Steven Spielberg'], 1999: ['Sam Mendes'], 2000: ['Steven Soderbergh'], 2001: ['Ron Howard'], 2002: ['Roman Polanski'], 2003: ['Peter Jackson'], 2004: ['Clint Eastwood'], 2005: ['Ang Lee'], 2006: ['Martin Scorsese'], 2007: ['Ethan Coen', 'Joel Coen'], 2008: ['Danny Boyle'], 2009: ['Kathryn Bigelow'], 2010: ['Tom Hooper']}

### 1A: Create dictionary with the count of Oscar nominations for each director 
nom_count_dict = {}
str1 = []
# Add your code here
for year, nom in nominated.items():
   for i in nom:
      nom_count_dict[i] = nom_count_dict.get(i, 0) + 1
print("nom_count_dict = {}\n".format(nom_count_dict))


### 1B: Create dictionary with the count of Oscar wins for each director
win_count_dict = {}
for year, winnerlist in winners.items():
   for winner in winnerlist:
      win_count_dict[winner] = win_count_dict.get(winner, 0) + 1
print("win_count_dict = {}\n".format(win_count_dict))



#FIRST PART OF SOLUTION
win_count_dict = {}
for year, winnerlist in winners.items():
   for winner in winnerlist:
      win_count_dict[winner] = win_count_dict.get(winner, 0) + 1

#SECOND PART OF SOLUTION
highest_count = 0
most_win_director = []
for key, value in win_count_dict.items():
   if value > highest_count:
      highest_count = value
      most_win_director.clear()
      most_win_director.append(key)
   elif value == highest_count:
      most_win_director.append(key)
   else:
      continue
print(most_win_director)


#ALTERNATIVE SECOND PART OF SOLUTION
most_win_director = [key for key, value in win_count_dict.items() if value == max(win_count_dict.values())]
print(most_win_director)




# Functions

def cylinder_volume(height, radius=5):
   pi = 3.14159
   return height * pi * radius ** 2
cylinder_volume(10)


# lambda finction

lam1 = lambda x, y=2: x ** y
lam1(3)

cities = ["New York City", "Los Angeles", "Chicago", "Mountain View", "Denver", "Boston"]
short_cities = list(filter(lambda name: len(name) < 10, cities))
print(short_cities)

from itertools import compress
cities = ["New York City", "Los Angeles", "Chicago", "Mountain View", "Denver", "Boston"]
tf = [True, True, False, False, False, False, False]
list(compress(cities, tf))


numbers = [
    [34, 63, 88, 71, 29],
    [90, 78, 51, 27, 45],
    [63, 37, 85, 46, 22],
    [51, 22, 34, 11, 18]
]
averages = list(map(lambda x: sum(x) / len(x), numbers))
print(averages)

cities = [["New York City", "Los Angeles"], ["Chicago", "Mountain View", "Denver"], ["Boston", "New York City", "Los Angeles", "Chicago"], ["Mountain View", "Denver", "Boston"]]
short_cities = list(map(lambda city: filter(lambda name: len(name) < 10, city), cities))
print(list(short_cities[1]))

for i in short_cities:
   for j in i:
      print(j)



# Scripting

# We can also interpret user input as a Python expression using the built-in function eval. 
# This function evaluates a string as a line of Python.
x = eval(input("Enter an expression: "))


names = input("get and process input for a list of names ").split(",")
assignments =  input("get and process input for a list of the number of assignments ").split(",")
grades =  input("get and process input for a list of grades ").split(",")
# message string to be used for each student
# HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to submit before you can \
   graduate. You're current grade is {} and can increase to {} if you submit all assignments before \
   the due date.\n\n"

#for i in range(len(names)):
#    print(message.format(names[i], assignments[i], grades[i], int(grades[i]) + 2 * int(assignments[i])))

for name, assignment, grade in zip(names, assignments, grades):
   print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))



# Syntax errors occur when Python can’t interpret our code, since we didn’t follow the correct syntax for Python. These are errors you’re likely to get when you make a typo, or you’re first starting to learn Python.
# Exceptions occur when unexpected things happen during execution of a program, even if the code is syntactically correct. There are different types of built-in exceptions in Python, and you can see which exception is thrown in the error message.

x = 1
while True:
   try:
      x = int(input("Enter a number: "))
      # print(x)
      break
   except ValueError:
      print("that\'s not a walid number\n")
   except KeyboardInterrupt:
      print("No onput tekken\n")
      break
   finally:
      print(x)
      print("zawsze\n")



# Reading and Writing Files
import os
print(os.path.expanduser("~"))
path = os.path.join(os.path.expanduser('~'), 'documents', 'python')
print (path)


# read a file
dire = "/home/ukasz/Documents/Programowanie/Python/udacity/"
f = open(dire+'udacity1.py', 'r')
file_data = f.read()
f.close()
print(file_data)


# write a file
f = open(dire+"test_write.py", "w")
f.write("write test")
f.close()


# open file many times
files = []
for i in range(10000):
   files.append(open(dire+'test_write.py', 'r'))
   print(i)


with open(dire+'udacity1.py', 'r') as f:
   file_data = f.read()
print(file_data)


with open(dire+"udacity1.py", "r") as snake:
   print(snake.read(5))
   print(snake.read(5))
   print(snake.read())


snake_lines = []
with open(dire+"udacity1.py", "r") as snake:
   for line in snake:
      snake_lines.append(line.strip())
print(snake_lines)


with open(dire+"udacity1.py", "r") as snake:
   list_of_lines = snake.readlines()
#print(list_of_lines)
for line in list_of_lines:
   print(line)


# You're going to create a list of the actors who appeared in the television programme Monty Python's Flying Circus.
dire = "/home/ukasz/Documents/Programowanie/Python/udacity/"
def create_cast_list(filename):
   cast_list = []
   #use with to open the file filename
   #use the for loop syntax to process each line
   #and add the actor name to cast_list
   with open(dire+"flying_circus_cast.txt", "r") as dirty_cast:
      for actor in dirty_cast:
         cast_list.append(actor.split(",")[0])
   return cast_list

cast_list = create_cast_list('flying_circus_cast.txt')
##for actor in cast_list:
##    print(actor)
print(cast_list)



# initiate empty list to hold user input and sum value of zero
user_list = []
list_sum = 0

# seek user input for ten numbers 
for _ in range(1):
    userInput = input("Enter any 2-digit number: ")
    
# check to see if number is even and if yes, add to list_sum
# print incorrect value warning  when ValueError exception occurs
    try:
        number = int(userInput)
        user_list.append(number)
        if number % 2 == 0:
            list_sum += number
    except ValueError:
        print("Incorrect value. That's not an int!")

print("user_list: {}".format(user_list))
print("The sum of the even numbers in user_list is: {}.".format(list_sum))
   

   
