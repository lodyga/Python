# content of variable/value
# variable name

SyntaxError
ValueError
AssertionError



command-line arguments: python3 name.py ... arguments
# run in sepatarte file 
import sys
print("hello", sys.argv)

# ASCII art
import cowsay
cowsay.pig("Moooooooooo")


# import songs data from apple store
import requests
import json

response = requests.get("https://itunes.apple.com/search?entity=song&limit=6&term=two_steps_from_hell")
# print(json.dumps(response.json(), indent=2))

for result in response.json()["results"]:
    print(result["trackName"])



import requests
import json

def main():
    # print(tracks(6))
    return tracks(6)

def tracks(n):
    response = requests.get("https://itunes.apple.com/search?entity=song&limit=" + str(n) + "&term=two_steps_from_hell")
    return [result["trackName"] for result in response.json()["results"]]

if __name__ == "__main__":
    print(main())

# importing from delme2.py
delme3.py

from .delme2 import tracks

print(tracks(50))


# tests
multi.py

def main():
    print(multi(int(input("Give a number: "))))

def multi(n):
    return n**2

if __name__ == "__main__":
    main()


test_multi.py

from multi import multi

def main():
    multi_test()

def multi_test():
    try:
        assert multi(3) == 4
    except AssertionError:
        print("2**2 should be 4")

if __name__ == "__main__":
    main()



# with pytest
test_multi.py

from multi import multi

def test_multi():
    assert multi(2) == 4
    assert multi(2) == 5
    assert multi(3) == 9

>pytest test_multi.py









# open .txt file
list_of_names = ["Ala", "Kot", "Daniel"]
# name = "Ala"


# with open("names.txt", "a") as file:
#     for name in list_of_names:
#         file.write(name+"\n")


with open("names.txt", "r") as file:
    #  for line in file.readlines():
    #     print(line.rstrip())
    for line in sorted(file, reverse=True):
        print(line.rstrip())





# read a csv file
with open("data.csv") as file:
    for line in sorted(file, key=lambda x: x.rstrip().split(",")[1]):
        lang, person, year, ext = line.rstrip().split(",")
        print(person, ext)



# read csv file with csv.reader
# better use csv.DictReader
import csv

with open("data.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)



# read csv file with csv.DictReader
# probably better use Pandas

import csv

languages = []

with open("data_names.csv") as file:
    reader = csv.DictReader(file)
    # Programming language, Designed by, Appeared, Extension
    for row in reader:
        languages.append({"Appeared": row["Appeared"], "Extension": row["Extension"]})
        # languages.append(row)
        # print(row["Programming language"], "created in", row["Appeared"], "by", row["Designed by"])

def sort_by(x):
    return x["Appeared"]    

for language in sorted(languages, key=sort_by):
    print(language)






# wirte csv as csv.witer
import csv

name = "Ukasz2"
home = "Wro Poalnd"

with open("write.txt", "a") as file:
    writer = csv.writer(file)
    writer.writerow([name, home])



# write csv as csv.DictWriter
import csv

name = "Ukasz3"
home = "Wro Poalnd3"

with open("write.txt", "a") as file:
    writer = csv.DictWriter(file, fieldnames=["name", "home"])
    writer.writerow({"name": name, "home": home})


# create a gif
import sys

from PIL import Image

images = []

for arg in sys.argv[1:]:
    image = Image.open(arg)
    images.append(image)

images[0].save(
    "costumes.gif", save_all=True, append_images=[images[1]], duration=200, loop=0
)






# regex search
import re

name = input("Input name: ")

if matches := re.search(r"^(\w+), *(\w+)$", name):
    # last, first = matches.groups()
    # name = f"{first} {last}"
    name = f"{matches.group(2)} {matches.group(1)}"

print(f"Hello, {name}")

# (?:...) don't capture, just group
# := walrus operator, assing a vlaue and ask bool question


re.sub(r"(https?://)?(www\.)?twitter\.com/", "", "https://twitter.com/ukasz")

matches =re.search(r"^(?:https?://)?(?:www\.)?twitter\.com/(.*)$", "https://twitter.com/ukasz", re.IGNORECASE)
matches.group(1)




procedural: from top to bottom, step by step
functional: pass functions around
object oriented programming: based on classes

# almost a class
class Student:
    ...

def main():
    student = get_student()
    print(f"Name: {student.name} from {student.home}")

def get_student():
    stu = Student()
    stu.name = input("Give a name: ")
    stu.home = input("Give a home: ")
    return stu

if __name__ == "__main__":
    main()

"""
attributes = instance variables = ._<some_name>
methods = function inside a class
__method__

property = secured attribute
decorator = function that modifies another function

variables for data
functions for action

instance of class if an object
"""


class Student:
    def __init__(self, name, house):
        # if not name:
        #     raise ValueError("Missing name")
        self.name = name
        self.house = house

    def __str__(self) -> str:
        return f"{self.name} from {self.house}"
    
    def walk(self):
        return 0 if self.house == "house" else 1
    
    # Getter
    @property
    def name(self):
        return self._name
    
    # Setter
    @name.setter
    def name(self, name):
        if not name:
            raise ValueError("Missing name")
        self._name = name

    # Getter
    @property
    def house(self):
        return self._house

    # Setter
    @house.setter
    def house(self, house):
        if house not in ["house", "home"]:
            raise ValueError("Invalid house")            
        self._house = house

def main():
    student = get_student()
    # student.house = "Not house"
    print(student)

def get_student():
    name = input("Give a name: ")
    house = input("Give a house: ")
    return Student(name, house)

if __name__ == "__main__":
    main()


"""
instance: {
    instance variables, 
    instance methods}

class: {
    class variables, 
    class methods @classmethod}

Instance methods doesn't have decortors.
Class methods have @classmethod
@staticmethod
"""

import numpy as np

class Hat:
    houses = ["G", "H", "R", "S"]
    
    def __init__(self):
        self.houses = ["Q"]  # overrides class attribute

    @classmethod
    def sort(cls, name):  # class method 
        print(f"{name} is in house {np.random.choice(cls.houses)}")
        print(f"{cls}")
    
    def sort2(self, name):  # instance method
        print(f"{name} is in house {np.random.choice(self.houses)}")
        print(f"{self}")

hat = Hat()
hat.sort("Harry")
hat.sort2("Merry")


# get students name within class not function
class Student:
    def __init__(self, name, house):
        self.name = name
        self.house = house

    def __str__(self) -> str:
        return f"{self.name} from {self.house}"
    
    def walk(self):
        return 0 if self.house == "house" else 1

    @classmethod
    def get(cls):
        name = input("Name: ")
        house = input("House: ")
        return cls(name, house)

def main():
    student_1 = Student.get()
    print(student_1.walk())
    print(student_1)

if __name__ == "__main__":
    main()



# inheritance
class Wizard:
    def __init__(self, name) -> None:
        if not name:
            raise ValueError("Missing name")
        self.name = name

    def __str__(self) -> str:
        return f"From Wizard class: {self.name}"

class Student(Wizard):
    def __init__(self, name, house) -> None:
        super().__init__(name)
        self.house = house
    
    def __str__(self) -> str:
        return f"From Student class: {self.name}"


class Professor(Wizard):
    def __init__(self, name, subject) -> None:
        super().__init__(name)
        self.subject = subject
    
    def __str__(self) -> str:
        return super().__str__() + f" with Proffessor class {self.name} and {self.subject}"


wizzard = Wizard("Albus")
student = Student("Harry", "G")
professor = Professor("Severuse", "Defense")

print(wizzard)
print(student)
print(professor)





# operator overloading
class Vault:
    def __init__(self, gallenons=0, sickles=0, knuts=0) -> None:
        self.gallenons = gallenons
        self.sickles = sickles
        self.knuts = knuts

    def __str__(self) -> str:
        return f"{self.gallenons} Gallenons, {self.sickles} Sickles, {self.knuts} Knuts"
    
    def __add__(self, other):
        gallenons = self.gallenons + other.gallenons
        sickles = self.sickles + other.sickles
        knuts = self.knuts + other.knuts
        return Vault(gallenons, sickles, knuts)




potter = Vault(100, 50, 25)
weasley = Vault(25, 50, 100)
# print(potter)

# gallenons = potter.gallenons + weasley.gallenons
# print(gallenons)

# total = Vault(gallenons, 100, 125)
# print(total)

total = potter + weasley
print(total)




students = [
    {"name": "H", "house": "G"},
    {"name": "R", "house": "G"},
    {"name": "D", "house": "S"},
]

set(student["house"] for student in students)

print(__name__)




class Account:
    def __init__(self) -> None:
        self._balance = 0
    
    @property
    def balance(self):
        return self._balance

    def deposiot(self, n):
        self._balance += n
    
    def withdraw(self, n):
        self._balance -= n

def main():
    account = Account()
    account.deposiot(100)
    account.withdraw(50)
    print(account.balance)


if __name__ == "__main__":
    main()
        
        
        
        
# class instance method
class Cat:
    MEOWS = 3
    
    def meow(self):
        for _ in range(Cat.MEOWS):
            print("meow")

# cat = Cat()
Cat().meow()


# class instance method without side effects
class Cat:
    MEOWS = 3
    
    def meow(self):
        return "meow\n" * self.MEOWS

Cat().meow()



# class method
class Cat:
    MEOWS = 3
    
    @classmethod
    def meow(cls):
        for _ in range(cls.MEOWS):
            print("meow")

Cat.meow()



# class method without side effects
class Cat:
    MEOWS = 3
    
    @classmethod
    def meow(cls):
        return "meow\n" * cls.MEOWS

Cat.meow()




# flags, switches handeling
# python3 <file-name.py> -n 3
import argparse

parser = argparse.ArgumentParser("Meow like a cat.")
parser.add_argument("-n", default=1, type=int, help="numbers of times to meow")
args = parser.parse_args()

for _ in range(args.n):
    print("meow")


# *
unpack: pass individual element of a list to a function
students = ["Mona", "Lisa", "Sting"]
print_middle = lambda a, b, c: print(b)  # catches only the middle element
print_middle(*students)

def total(gallenons, sickles, knuts):
    return ((gallenons  * 17) + sickles) * 29 + knuts

coins_list = [100, 50, 25]

# unpack a list
total(*coins_list)


coins_dict = {"gallenons":100, "sickles":50, "knuts":25}
# unpack a dict
total(**coins_dict)
# unpacked dict data provided
total(gallenons=100, sickles=50, knuts=25)



def f(*args, **kwargs):
    print("Positional: ", args)
    print("Named: ", kwargs)

f(100, 50, 25,)  # Positional:  (100, 50, 25)
f(*(100, 50, 25,))  # Positional:  (100, 50, 25)
f(gallenons=100, sickles=50, knuts=25,)  # Named:  {'gallenons': 100, 'sickles': 50, 'knuts': 25}
f(**{"gallenons":100, "sickles":50, "knuts":25})  # Named:  {'gallenons': 100, 'sickles': 50, 'knuts': 25}
f(100, 50, 25, gallenons=100, sickles=50, knuts=25,)  # Positional:  (100, 50, 25)  Named:  {'gallenons': 100, 'sickles': 50, 'knuts': 25}



# functional progammning with map
map, lambda, filter

def main():
    print(*yell("ala", "ma", "kota."))

def yell(*words):
    return [word.upper() for word in words]

main()



def main():
    print(*yell("ala", "ma", "kota."))

def yell(*words):
    return map(str.upper, words)

main()




# "".join(list(map(str.upper, "ala ma kota.")))
print(*map(str.upper, "ala ma kota."))
print(*map(str.upper, ("ala", "ma", "kota.")))




# filter

students = [
    {"name": "H", "house": "G"},
    {"name": "R", "house": "G"},
    {"name": "D", "house": "S"},
]

# def is_gr(s):
#     return s["house"] == "G"

is_gr = lambda s: s["house"] == "G"


print(*filter(is_gr, students))
print(*filter(lambda s: s["house"] == "G", students))







# gerenator
# yield

def main():
    n = int(input("Give a numeber: "))
    for s in sheep(n):
        print(s)

# without generator
def sheep(n):
    return ["Δ"*i for i in range(n)]

# with generator
def sheep(n):
    for i in range(n):
        yield "Δ" * i


main()




import cowsay
import pyttsx3

engine = pyttsx3.init()
this = input("What's up?: ")
cowsay.cow(this)
engine.say(this)
engine.runAndWait()

