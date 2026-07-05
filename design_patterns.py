# {design patterns: {structural, behavioral, creational}}

# structural pattern
# factory method
# Defines an interface for creating an object in a superclass, but lets subclasses decide which class to instantiate (to alter the type of objects that will be created).


from abc import ABC, abstractmethod

# Product
# Step 1: Define the Product Interface
class Notification(ABC):
    @abstractmethod
    def send(self, message: str) -> str:
        pass

# Concrete Product
# Step 2: Create Concrete Products
class EmailNotification(Notification):
    def send(self, message: str) -> str:
        return f"Sending Email: {message}"

class SMSNotification(Notification):
    def send(self, message: str) -> str:
        return f"Sending SMS: {message}"

# Creator
# Step 3: Define the Creator Class
class NotificationFactory(ABC):
    # factory method
    @abstractmethod
    def create_notification(self) -> Notification:
        pass

    def some_operation(self, message: str) -> str:
        notification = self.create_notification()
        result = f"Some operation {notification.send(message)}"
        return result

# Concrete Creator
# Step 4: Implement Concrete Creators
class EmailNotificationFactory(NotificationFactory):
    def create_notification(self) -> Notification:
        return EmailNotification()

class SMSNotificationFactory(NotificationFactory):
    def create_notification(self) -> Notification:
        return SMSNotification()

# Client Code
def send_notification(factory: NotificationFactory, message: str) -> str:
    notification = factory.create_notification()
    return notification.send(message)

# Client Code 2
def send_notification2(factory: NotificationFactory, message: str) -> str:
    notification = factory.some_operation(message)
    return notification


# Example Usage
# if __name__ == "__main__":
email_factory = EmailNotificationFactory()
sms_factory = SMSNotificationFactory()
    
# print(send_notification(email_factory, "Hello via Email!"))
# print(send_notification(sms_factory, "Hello via SMS!"))
print(send_notification(email_factory, "Hello via Email!"))








from abc import ABC, abstractmethod
from numpy.random import choice

# Step 1: Define the Product Interface
class Animal(ABC):
    @abstractmethod
    def make_sound(self) -> None:
        pass

# Step 2: Create Concrete Products
class Giraffe(Animal):
    def make_sound(self, origin: str) -> None:
        print(f"Giraffe: *gentle hum*, {origin}")

class Deer(Animal):
    def make_sound(self, origin: str) -> None:
        print(f"Deer: *bleat*, {origin}")

class Wolf(Animal):
    def make_sound(self, origin) -> None:
        print(f"Wolf: *howl*, {origin}")

# Step 3: Define the Abstract Creator
class AnimalFactory(ABC):
    @abstractmethod
    # factory method
    def create_animal(self) -> Animal:
        pass

# Step 4: Implement Concrete Creators
class ForestAnimalFactory(AnimalFactory):
    # factory method
    def create_animal(self) -> Animal:
        # Forest animals: Wolf or Deer
        return choice([Wolf(), Deer()])

class PlainsAnimalFactory(AnimalFactory):
    # factory method
    def create_animal(self) -> Animal:
        # Plains animals: Giraffe or Deer
        return choice([Giraffe(), Deer()])

# Client Code
def introduce_animal(factory: AnimalFactory, origin: str) -> None:
    animal = factory.create_animal()
    animal.make_sound(origin)

# Example Usage
# if __name__ == "__main__":
forest_factory = ForestAnimalFactory()
plains_factory = PlainsAnimalFactory()

print("Forest Animal:")
introduce_animal(forest_factory, "a forest animal.")
print("\nPlains Animal:")
introduce_animal(plains_factory, "a plains animal.")





class Burger(ABC):
    def __init__(self, ingredients: list[str]):
        self.indegrients = ingredients

    @property
    def ingredients(self):
        print(self.indegrients)

class BurgerFactory(ABC):
    def createCheeseBurger(self):
        ingredients = ["bun", "cheese", "grilled-meat"]
        return Burger(ingredients)

burgerFactory = BurgerFactory()
cheeseBurger = burgerFactory.createCheeseBurger()
cheeseBurger.ingredients
print(cheeseBurger)













# structural pattern
# abstract factory
# Provides an interface for creating families of related or dependent objects without specifying their concrete classes.
# Factory can produce different products.


from abc import ABC, abstractmethod


# ===== Abstract Products =====

class VirtualMachine(ABC):
    @abstractmethod
    def start(self):
        pass


class Database(ABC):
    @abstractmethod
    def connect(self):
        pass


class Storage(ABC):
    @abstractmethod
    def upload(self, filename):
        pass


# ===== AWS Products =====

class AWSVirtualMachine(VirtualMachine):
    def start(self):
        print("Starting AWS EC2 instance")


class AWSDatabase(Database):
    def connect(self):
        print("Connecting to AWS RDS")


class AWSStorage(Storage):
    def upload(self, filename):
        print(f"Uploading {filename} to AWS S3")


# ===== Azure Products =====

class AzureVirtualMachine(VirtualMachine):
    def start(self):
        print("Starting Azure Virtual Machine")


class AzureDatabase(Database):
    def connect(self):
        print("Connecting to Azure SQL Database")


class AzureStorage(Storage):
    def upload(self, filename):
        print(f"Uploading {filename} to Azure Blob Storage")


# ===== Abstract Factory =====

class CloudFactory(ABC):
    @abstractmethod
    def create_vm(self) -> VirtualMachine:
        pass

    @abstractmethod
    def create_database(self) -> Database:
        pass

    @abstractmethod
    def create_storage(self) -> Storage:
        pass


# ===== Concrete Factories =====

class AWSFactory(CloudFactory):
    def create_vm(self) -> VirtualMachine:
        return AWSVirtualMachine()

    def create_database(self) -> Database:
        return AWSDatabase()

    def create_storage(self) -> Storage:
        return AWSStorage()


class AzureFactory(CloudFactory):
    def create_vm(self) -> VirtualMachine:
        return AzureVirtualMachine()

    def create_database(self) -> Database:
        return AzureDatabase()

    def create_storage(self) -> Storage:
        return AzureStorage()


# ===== Client =====

class DeploymentManager:
    def __init__(self, factory: CloudFactory) -> None:
        self.vm = factory.create_vm()
        self.db = factory.create_database()
        self.storage = factory.create_storage()

    def deploy(self):
        self.vm.start()
        self.db.connect()
        self.storage.upload("app.zip")


# ===== Usage =====

factory = AWSFactory()
deployment = DeploymentManager(factory)
deployment.deploy()

print()

factory = AzureFactory()
deployment = DeploymentManager(factory)
deployment.deploy()

















# creational pattern
# builder
# Builder is a creational design pattern that lets you construct complex objects step by step. The pattern allows you to produce different types and representations of an object using the same construction code.

from abc import ABC, abstractmethod
from typing import Any


class Product1():
    def __init__(self) -> None:
        self.parts = []

    def add(self, part: Any) -> None:
        self.parts.append(part)

    def list_parts(self) -> None:
        print(f"Product parts: {', '.join(self.parts)}", end="")


class Builder(ABC):
    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def produce_part_a(self) -> None:
        pass

    @abstractmethod
    def produce_part_b(self) -> None:
        pass

    @abstractmethod
    def produce_part_c(self) -> None:
        pass


class ConcreteBuilder1(Builder):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._product = Product1()

    @property
    def product(self) -> Product1:
        product = self._product
        self.reset()
        return product

    def produce_part_a(self) -> None:
        self._product.add("PartA1")

    def produce_part_b(self) -> None:
        self._product.add("PartB1")

    def produce_part_c(self) -> None:
        self._product.add("PartC1")


class Director:
    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        self._builder = builder

    def build_minimal_viable_product(self) -> None:
        self.builder.produce_part_a()

    def build_full_featured_product(self) -> None:
        self.builder.produce_part_a()
        self.builder.produce_part_b()
        self.builder.produce_part_c()


if __name__ == "__main__":
    director = Director()
    builder = ConcreteBuilder1()
    director.builder = builder

    print("Standard basic product: ")
    director.build_minimal_viable_product()
    builder.product.list_parts()

    print("\n")

    print("Standard full featured product: ")
    director.build_full_featured_product()
    builder.product.list_parts()

    print("\n")

    # Remember, the Builder pattern can be used without a Director class.
    print("Custom product: ")
    builder.produce_part_a()
    builder.produce_part_b()
    builder.product.list_parts()
    print("\n")





"""
Structure of the Builder pattern
Product – the complex object being created (Computer).
Builder – defines the steps required to build the product (ComputerBuilder).
Concrete Builder – implements those steps (GamingComputerBuilder, OfficeComputerBuilder).
Director – orchestrates the building process (ComputerDirector).
Client – chooses which builder to use.
"""

from abc import ABC, abstractmethod


# Product
class Computer:
    def __init__(self):
        self.cpu = None
        self.ram = None
        self.storage = None
        self.gpu = None

    def __str__(self):
        return (
            f"Computer("
            f"cpu={self.cpu}, "
            f"ram={self.ram}, "
            f"storage={self.storage}, "
            f"gpu={self.gpu})"
        )


# Builder interface
class ComputerBuilder(ABC):
    @abstractmethod
    def build_cpu(self):
        pass

    @abstractmethod
    def build_ram(self):
        pass

    @abstractmethod
    def build_storage(self):
        pass

    @abstractmethod
    def build_gpu(self):
        pass

    @abstractmethod
    def get_result(self):
        pass


# Concrete builder
class GamingComputerBuilder(ComputerBuilder):
    def __init__(self):
        self.computer = Computer()

    def build_cpu(self):
        self.computer.cpu = "Intel Core i9"

    def build_ram(self):
        self.computer.ram = "32 GB"

    def build_storage(self):
        self.computer.storage = "2 TB SSD"

    def build_gpu(self):
        self.computer.gpu = "RTX 5090"

    def get_result(self):
        return self.computer


# Another concrete builder
class OfficeComputerBuilder(ComputerBuilder):
    def __init__(self):
        self.computer = Computer()

    def build_cpu(self):
        self.computer.cpu = "Intel Core i3"

    def build_ram(self):
        self.computer.ram = "8 GB"

    def build_storage(self):
        self.computer.storage = "512 GB HDD"

    def build_gpu(self):
        self.computer.gpu = "Integrated Graphics"

    def get_result(self):
        return self.computer


# Director
class ComputerDirector:
    def construct(self, builder: ComputerBuilder):
        builder.build_cpu()
        builder.build_ram()
        builder.build_storage()
        builder.build_gpu()
        return builder.get_result()


# Client code
director = ComputerDirector()

gaming_pc = director.construct(GamingComputerBuilder())
office_pc = director.construct(OfficeComputerBuilder())

print(gaming_pc)
print(office_pc)

builder = GamingComputerBuilder()
builder.build_gpu()
print(builder.get_result())




# Fluent builder
class Computer:
    def __init__(self):
        self.cpu = None
        self.ram = None
        self.storage = None
        self.gpu = None

    def __str__(self):
        return (
            f"Computer(cpu={self.cpu}, "
            f"ram={self.ram}, "
            f"storage={self.storage}, "
            f"gpu={self.gpu})"
        )


class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()

    def cpu(self, value):
        self.computer.cpu = value
        return self

    def ram(self, value):
        self.computer.ram = value
        return self

    def storage(self, value):
        self.computer.storage = value
        return self

    def gpu(self, value):
        self.computer.gpu = value
        return self

    def build(self):
        return self.computer


pc = (
    ComputerBuilder()
    .cpu("AMD Ryzen 9")
    .ram("64 GB")
    .storage("4 TB SSD")
    .gpu("RTX 5080")
    .build()
)

print(pc)






class Burger:
    def __init__(self):
        self.buns = None
        self.patty = None
        self.cheese = None
    
    def set_buns(self, bun_style):
        self.buns = bun_style

    def set_patty(self, patty_style):
        self.patty = patty_style
    
    def set_cheese(self, cheese_style):
        self.cheese = cheese_style

burger = Burger()
burger.set_buns("wholewheat")
burger.set_cheese("gouda")
print(burger.buns)

class BurgerBuilder:
    def __init__(self):
        self.burger = Burger()
    
    def add_buns(self, bun_style):
        self.burger.set_buns(bun_style)
        return self
    
    def add_patty(self, patty_style):
        self.burger.set_patty(patty_style)
        return self
    
    def add_cheese(self, cheese_style):
        self.burger.set_cheese(cheese_style)
        return self
    
    def build(self):
        return self.burger

burger = BurgerBuilder() \
    .add_buns("white") \
    .add_patty("all-in") \
    .add_cheese("gouda") \
    .build()

# print(burger.buns)

class BurgerDirector:
    def build_cheeseburger(self, builder: BurgerBuilder):
        builder \
            .add_buns("bun") \
            .add_patty("patty") \
            .add_cheese("quatro")

director = BurgerDirector()
cheesburger = BurgerBuilder()
director.build_cheeseburger(cheesburger)
burger = cheesburger.build()
























# creational pattern
# singleton
# a class has only one instance, while providing a global access point to this instance.

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    def some_business_logic(self):
        """
        Finally, any singleton should define some business logic, which can be
        executed on its instance.
        """

        # ...


if __name__ == "__main__":
    # The client code.

    s1 = Singleton()
    s2 = Singleton()

    if id(s1) == id(s2):
        print("Singleton works, both variables contain the same instance.")
    else:
        print("Singleton failed, variables contain different instances.")







class ApplicationState:
    instance = None

    def __init__(self):
        self.is_loggedin = False
    
    @staticmethod
    def get_instance():
        if not ApplicationState.instance:
            ApplicationState.instance = ApplicationState()

        return ApplicationState.instance


app_state_1 = ApplicationState().get_instance()
print(app_state_1.is_loggedin)  # False

app_state_2 = ApplicationState().get_instance()
print(app_state_2.is_loggedin)  # False

app_state_1.is_loggedin = True

print(app_state_1.is_loggedin)  # True
print(app_state_2.is_loggedin)  # True


















# structural pattern
# facade
# Higher level unified interface for a set of lower level (interfaces) classes. 
# Dynamic arrays == resize static arrays.













# structural pattern
# adapter (wrapper)
# Converts interface (of a class) to another interface (that client expects).

# Interface tagret
class UsbCable:
    def __init__(self):
        self.is_plugged = False

    def plug_usb(self):  # request
        self.is_plugged = True


class UsbPort:
    def __init__(self):
        self.is_port_avaible = True

    def plug(self, usb_cable):
        if self.is_port_avaible:
            usb_cable.plug_usb()
            self.is_port_avaible = False

# UsbCable can plug directly into UsbPort
usb_cable = UsbCable()  # new cable
usb_port_1 = UsbPort()  # new port
usb_cable.is_plugged  # False
usb_port_1.is_port_avaible  # True
usb_port_1.plug(usb_cable)  # plug usb cable into usb port
usb_cable.is_plugged  # True
usb_port_1.is_port_avaible  # False


# Adaptee
class MicroUsbCable:
    def __init__(self):
        self.is_plugged = False
    
    def plug_micro_usb(self):  # specific request
        self.is_plugged = True


# Adapter
class MicroToUsbAdapter(UsbCable):
    def __init__(self, micro_usb_cable):
        self.micro_usb_cable = micro_usb_cable  # adaptee
        self.micro_usb_cable.plug_micro_usb()  # request method


# can override UsbCable.plug_usb()
micro_to_usb_adapter = MicroToUsbAdapter(MicroUsbCable())
usb_port_2 = UsbPort()
usb_port_2.is_port_avaible  # True
usb_port_2.plug(micro_to_usb_adapter)
micro_to_usb_adapter.is_plugged  # True
usb_port_2.is_port_avaible  # False
























# structural pattern
# proxy: remote, virtual, protection
# Provides a placeholder for another object in order to get access to it.


# virtual
from abc import ABC, abstractmethod
from time import sleep

# Step 1: Define the Subject interface (common interface for RealImage and Proxy)
class Image(ABC):
    @abstractmethod
    def display(self):
        pass

# Step 2: Implement RealSubject (RealImage)
class RealImage(Image):
    def __init__(self, filename):
        self.filename = filename
        self.load_from_disk()

    def load_from_disk(self):
        print(f"Loading image: {self.filename}")
        sleep(2)  # Simulating a time-consuming operation

    def display(self):
        print(f"Displaying image: {self.filename}")

# Step 3: Implement Proxy (ImageProxy)
class ImageProxy(Image):
    def __init__(self, filename):
        self.filename = filename
        self.real_image = None  # Lazy initialization

    def display(self):
        if self.real_image is None:
            self.real_image = RealImage(self.filename)  # Load only when needed
        self.real_image.display()

# Client Code
image1 = ImageProxy("photo1.jpg")

# Image is not loaded yet
print("Doing something else before displaying the image...")

# Image is now loaded only when needed
image1.display()

# Subsequent calls do not reload the image
image1.display()






# protection
from abc import ABC, abstractmethod

# Step 1: Define the Subject interface
class BankAccount(ABC):
    @abstractmethod
    def withdraw(self, amount):
        pass

# Step 2: Implement the Real Subject (Actual Bank Account)
class RealBankAccount(BankAccount):
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            print(f"Insufficient funds! Balance: {self.balance}")
        else:
            self.balance -= amount
            print(f"Withdrawn {amount}. New balance: {self.balance}")

# Step 3: Implement the Proxy (Controls access)
class BankAccountProxy(BankAccount):
    def __init__(self, real_account, user):
        self.real_account = real_account
        self.user = user

    def withdraw(self, amount):
        if self.user == self.real_account.owner:
            self.real_account.withdraw(amount)
        else:
            print("Access Denied: You are not the account owner!")

# Client Code
real_account = RealBankAccount("Alice", 1000)
proxy_account = BankAccountProxy(real_account, "Bob")  # Unauthorized user

proxy_account.withdraw(200)  # Access Denied
proxy_account = BankAccountProxy(real_account, "Alice")  # Authorized user
proxy_account.withdraw(200)  # Successful withdrawal





# remote
import time
import random
from abc import ABC, abstractmethod

# Step 1: Define the Subject interface
class Database(ABC):
    @abstractmethod
    def fetch_data(self):
        pass

# Step 2: Implement the Real Subject (Remote Database)
class RemoteDatabase(Database):
    def fetch_data(self):
        time.sleep(.2)  # Simulating network delay
        return f"Data from remote server (ID: {random.randint(1000, 9999)})"

# Step 3: Implement the Proxy (Handles caching and delays)
class DatabaseProxy(Database):
    def __init__(self):
        self.real_db = RemoteDatabase()
        self.cache = None

    def fetch_data(self):
        if self.cache is None:
            print("Fetching from remote server...")
            self.cache = self.real_db.fetch_data()
        else:
            print("Returning cached data...")
        return self.cache

# Client Code
proxy_db = DatabaseProxy()

print(proxy_db.fetch_data())  # First call - fetches from remote
print(proxy_db.fetch_data())  # Second call - returns cached data

















# structural pattern
# decorator
# Attaches additional responsibilities to an object dynamically.


# Class-Based Decorator (Standard Approach)
from abc import ABC, abstractmethod

# Step 1: Define the Subject interface (Component)
class Coffee(ABC):
    @abstractmethod
    def cost(self):
        pass

    @abstractmethod
    def description(self):
        pass

# Step 2: Implement the Concrete Component (Real Object)
class SimpleCoffee(Coffee):
    def cost(self):
        return 5  # Base price

    def description(self):
        return "Simple Coffee"

# Step 3: Create an Abstract Decorator
class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee  # Composition: has a Coffee instance

    def cost(self):
        return self._coffee.cost()

    def description(self):
        return self._coffee.description()

# Step 4: Concrete Decorators that extend functionality
class MilkDecorator(CoffeeDecorator):
    def __init__(self, coffee):
        super().__init__(coffee)
        self.milk_cost = 1

    def cost(self):
        return self._coffee.cost() + self.milk_cost  # Adding milk costs extra

    def description(self):
        return self._coffee.description() + ", with Milk"

class SugarDecorator(CoffeeDecorator):
    def __init__(self, coffee):
        super().__init__(coffee)
        self.sugar_cost = 1
        
    def cost(self):
        return self._coffee.cost() + self.sugar_cost  # Adding sugar costs extra

    def description(self):
        return self._coffee.description() + ", with Sugar"

# Client Code
coffee = SimpleCoffee()
print(f"{coffee.description()} - ${coffee.cost()}")

coffee = MilkDecorator(coffee)  # Adding milk
print(f"{coffee.description()} - ${coffee.cost()}")

coffee = SugarDecorator(coffee)  # Adding sugar
print(f"{coffee.description()} - ${coffee.cost()}")






# Function Decorators (Pythonic Way)
def icing(func):
    def wrap(*args, **kwargs):
        return (
            "Sprinkles added\n" +
            func(*args, **kwargs))
    return wrap


def fudge(func):
    def wrap(*args, **kwargs):
        return (
            "Fudge added \n" +
            func(*args, **kwargs))
    return wrap


@icing
@fudge
def get_ice_cream(flavor="Normal"):
    return flavor + " ice"


print(get_ice_cream("Vanilla"))
print(get_ice_cream())















# behavioral pattern
# strategy
# Defines a family of algorithms encapsulate each one and makes them interchangable.
# Modify class withou changing it.
# open-close principle. Open for extension, closed for modifications.

from abc import ABC, abstractmethod

class FilterStrategy(ABC):
    @abstractmethod
    def remove_value(self, val):
        pass

class RemoveNegativesStrategy(FilterStrategy):
    def remove_value(self, val):
        return val < 0
    
class RemoveOddStrategy(FilterStrategy):
    def remove_value(self, val):
        return val % 2

class Values:
    def __init__(self, vals):
        self.vals = vals
    
    def filter(self, strategy):
        result = []

        for value in self.vals:
            if not strategy.remove_value(value):
                result.append(value)
        
        return result

values = Values([-7, -4, -1, 0, 2, 6, 9])
print(values.filter(RemoveNegativesStrategy()))  # [0, 2, 6, 9]
print(values.filter(RemoveOddStrategy()))  # [-4, 0, 2, 6]




# from typing import Protocol
from abc import ABC, abstractmethod

# Define the strategy interface using an abstract base class
class PaymentStrategy(ABC):
    @abstractmethod
    def pay_by(self, amount: float) -> None:
        pass

# Define the strategy interface
# class PaymentStrategy(Protocol):
#     def pay_by(self, amount: float) -> None:
#         ...

# Concrete strategy 1
class CreditCardPayment:
    def pay_by(self, amount: float) -> None:
        commisssion = amount * 1.02
        return (commisssion, "Credit Card.")

# Concrete strategy 2
class PayPalPayment:
    def pay_by(self, amount: float) -> None:
        commisssion = amount * 1.01
        return (commisssion, "PayPal.")

# Context class
class ShoppingCart:
    def __init__(self, payment_strategy: PaymentStrategy):
        self.payment_strategy = payment_strategy

    def checkout(self, amount: float) -> None:
        total_price, payment_type = self.payment_strategy.pay_by(amount)
        return f"Paid ${total_price} using ${payment_type}."

# Usage
cart1 = ShoppingCart(CreditCardPayment())
print(cart1.checkout(100))

cart2 = ShoppingCart(PayPalPayment())
print(cart2.checkout(200))
















# behavioral patterns
# observer
# Defines one to many dependency between objects so that when one object changet state all of its dependents are notified and updated automatically.
class YoutubeChannel():
    def __init__(self, name):
        self.name = name
        self.subscribers = set()
    
    def subscribe(self, sub):
        self.subscribers.add(sub)
    
    def notify(self, event):
        for sub in self.subscribers:
            sub.send_notification(self.name, event)

    def show_subscribers(self):
        return self.subscribers
    

from abc import ABC, abstractmethod

class YoutubeSubscriber(ABC):
    @abstractmethod
    def send_notification(self, event):
        pass

class YoutubeUser(YoutubeSubscriber):
    def __init__(self, name):
        self.name = name

    def send_notification(self, channel,  event):
        print(f"User {self.name} revieved notifiaction from {channel}: {event}.")


channel = YoutubeChannel("neetcode")
channel.subscribe(YoutubeUser("sub1"))
channel.subscribe(YoutubeUser("sub2"))
channel.subscribe(YoutubeUser("sub3"))
# print(channel.subscribers)
# channel.show_subscribers()
channel.notify("A new video released.")











# behavioral pattern
# iterator
# Provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

class ListNode:
    def __init__(self, val=float("-inf"), next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self, head):
        self.head = head
        self.node = None

    # define iterator
    def __iter__(self):
        self.node = self.head
        return self

    # iterate
    def __next__(self):
        if self.node:
            val = self.node.val
            self.node = self.node.next
            return val
        else:
            raise StopIteration

# initailize LinkedList
head = ListNode(2)
head.next = ListNode(4)
head.next.next = ListNode(5)
linked_list_1 = LinkedList(head)

# iterate through LinkedList
for node in linked_list_1:
    print(node)


node_3 = ListNode(9)
node_2 = ListNode(8, node_3)
node_1 = ListNode(7, node_2)
linked_list_2 =LinkedList(node_1)

# iterate through LinkedList
for node in linked_list_2:
    print(node)












# behavioral pattern
# command
# Encapsulates a request as an object therby letting parametrize other objects with different requests, queues or log requests.

