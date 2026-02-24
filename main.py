#1.Przygotować klasę Employee, która będzie przechowywać
# atrybuty: first_name, last_name i salary.
# Dodać metodę get_full_name(), zwracającą pełne imię i nazwisko.
# Następnie utworzyć klasę Manager, dziedziczącą po Employee,
# dodającą department oraz metodę get_department_info(),
# zwracającą informację o zarządzanym dziale.

class Employee:
    first_name: str
    last_name: str
    salary: float

    def __init__(self, first_name: str, last_name: str, salary: float) -> None:
        self.first_name = first_name
        self.last_name = last_name
        self.salary = salary

    def get_full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

class Manager(Employee):
    department: str

    def __init__(self, department: str) -> None:
        self.department = department

    def get_department_info(self):
        return f"{self.department}"

#2. Utworzyć klasę Transaction jako namedtuple zawierającą
# transaction_id, amount oraz currency.
# Następnie zdefiniować klasę BankAccount,
# która będzie miała atrybut balance oraz metodę apply_transaction(),
# przyjmującą obiekt Transaction i modyfikującą saldo.

from collections import namedtuple

Transaction = namedtuple("Transaction", ["transaction_id", "amount", "currency"])

class BankAccount:
    balance: float
    def __init__(self, balance: float):
        self.balance = balance

    def apply_transaction(self, transaction: Transaction):
        if transaction.amount <= 0:
            raise ValueError("amount must be greater than 0")
        self.balance += transaction
        return self.balance

#3. Napisać klasę Book używając dataclass, która zawiera
# title, author, year, price. Dodaj metodę apply_discount(),
# która obniży cenę książki o podany procent.

from dataclasses import dataclass, field

@dataclass(frozen=True)
class Book:
    title: str
    author: str
    year: str
    price: float

    def apply_discount(self, discount: float):
        if discount <= 0:
            raise ValueError("discount must be greater than 0")
        self.price -= self.price * discount


#4. Stworzyć klasę Product jako dataclass zawierającą name,
# price, category, a następnie rozszerz ją o walidację ceny
# (powinna być większa od zera) oraz domyślną wartość category="General".

@dataclass(frozen=True)
class Product:
    name: str
    price: float
    category: str

    def validate_price(self):
        if self.price > 0:
            return True
        return False


#5.Utworzyć klasę Car z atrybutami brand, model i year.
# Następnie dodać metodę is_classic(), która zwróci True,
# jeśli samochód ma ponad 25 lat.

@dataclass(frozen=True)
class Car:
    brand: str
    model: str
    year: int

    def is_classic(self):
        if 2026 - self.year > 25:
            return True
        return False

# 6. Stworzyć klasy ElectricVehicle oraz GasolineVehicle,
# które mają metodę fuel_type(), zwracającą odpowiednio "electric" i "gasoline".
# Następnie utworzyć klasę HybridCar, która dziedziczy po obu
# i nadpisuje metodę fuel_type(), aby zwracała "hybrid".

class ElectricVehicle:
    def fuel_type(self) -> str:
        return "electric"
class GasolineVehicle:
    def fuel_type(self) -> str:
        return "gasoline"

class HybridCar(ElectricVehicle, GasolineVehicle):
    def fuel_type(self) -> str:
        return "hybrid"

# 7. Utworzyć klasę Person z metodą introduce(),
# zwracającą "I am a person". Następnie stworzyć klasy Worker i Student,
# które dziedziczą po Person i zmieniają tę metodę na "I am a worker"
# oraz "I am a student". Następnie utworzyć klasę WorkingStudent,
# która dziedziczy zarówno po Worker, jak i Student, i sprawdź,
# jak Python rozwiąże konflikt metod.

class Person:
    def introduce(self) -> str:
        return "I am a person"

class Worker(Person):
    def introduce(self) -> str:
        return "I am a worker"

class Student(Person):
    def introduce(self) -> str:
        return "I am student"

class WorkingStudent(Worker, Student):
    pass


# Utworzyć klasy Animal i Pet. Klasa Animal powinna mieć metodę make_sound(),
# zwracającą "Some sound", a Pet powinna mieć metodę is_domestic(),
# zwracającą True. Następnie utworzyć klasę Dog, dziedziczącą po obu,
# i dostosować metody tak, aby pasowały do psa.

class Animal:
    def make_sound(self) -> str:
        return "some sound"

class Pet:
    def is_domestic(self):
        return True

class Dog(Animal, Pet):
    def is_domestic(self):
        return True

    def make_sound(self) -> str:
        return "hau hau"

# 9. Zaimplementować klasy FlyingVehicle i WaterVehicle,
# które mają metody move(), zwracające odpowiednio "I fly" oraz "I sail".
# Następnie stworzyć klasę AmphibiousVehicle, która łączy obie
# i pozwala na wybór trybu działania.

class FlyingVehicle:
    def move(self) -> str:
        return "I fly"

class WaterVehicle:
    def move(self) -> str:
        return "I sail"

class AmphibiousVehicle(FlyingVehicle, WaterVehicle):
    mode: object
    def move(self) -> str:
        if self.mode == FlyingVehicle:
            return "I fly"
        if self.mode == WaterVehicle:
            return "I sail"

# 10. Utworzyć klasę Robot z metodą operate(),
# zwracającą "Performing task", oraz AI z metodą think(),
# zwracającą "Processing data". Następnie utworzyć klasę Android,
# która dziedziczy po obu i dodaje własną metodę self_learn().

class Robot:
    def operate(self) -> str:
        return "Perfoming task"

class AI:
    def think(self) -> str:
        return "Processing data"

class Android(AI, Robot):
    def self_learn(self) -> str:
        return "Learning data"

# 11. Stworzyć klasę TemperatureConverter,
# która będzie zawierać metody statyczne celsius_to_fahrenheit()
# oraz fahrenheit_to_celsius().

class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(c: float):
        return c * 9 / 5 + 32

    @staticmethod
    def fahrenheit_to_celsius(f: float):
        return (f - 32) * 5/9

# 12. Przygotować klasę IDGenerator z metodą klasową generate_id(),
# która automatycznie generuje unikalne identyfikatory dla obiektów.
# Każdy nowo utworzony obiekt powinien otrzymać kolejny numer ID.

class IDGenerator:
    def generate_id(self) -> int:
        





