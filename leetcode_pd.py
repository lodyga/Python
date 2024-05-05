# 175. Combine Two Tables
# https://leetcode.com/problems/combine-two-tables/
"""
Write a solution to report the first name, last name, city, and state of each person in the Person table. If the address of a personId is not present in the Address table, report null instead.

Return the result table in any order.

Input: 
Person table:
+----------+----------+-----------+
| personId | lastName | firstName |
+----------+----------+-----------+
| 1        | Wang     | Allen     |
| 2        | Alice    | Bob       |
+----------+----------+-----------+
Address table:
+-----------+----------+---------------+------------+
| addressId | personId | city          | state      |
+-----------+----------+---------------+------------+
| 1         | 2        | New York City | New York   |
| 2         | 3        | Leetcode      | California |
+-----------+----------+---------------+------------+

Output: 
+-----------+----------+---------------+----------+
| firstName | lastName | city          | state    |
+-----------+----------+---------------+----------+
| Allen     | Wang     | Null          | Null     |
| Bob       | Alice    | New York City | New York |
+-----------+----------+---------------+----------+
"""

import pandas as pd

def combine_two_tables(person: pd.DataFrame, address: pd.DataFrame) -> pd.DataFrame:
    sol_df = pd.merge(person, address, how="left", on=["personId"])
    return sol_df[["firstName", "lastName", "city", "state"]]
    # return person.merge(address, how="left", on=["personId"])[["firstName", "lastName", "city", "state"]]





# 181. Employees Earning More Than Their Managers
# https://leetcode.com/problems/employees-earning-more-than-their-managers/description/
"""
Write a solution to find the employees who earn more than their managers.

Return the result table in any order.

The result format is in the following example.


Input: 
Employee table:
+----+-------+--------+-----------+
| id | name  | salary | managerId |
+----+-------+--------+-----------+
| 1  | Joe   | 70000  | 3         |
| 2  | Henry | 80000  | 4         |
| 3  | Sam   | 60000  | Null      |
| 4  | Max   | 90000  | Null      |
+----+-------+--------+-----------+
Output: 
+----------+
| Employee |
+----------+
| Joe      |
+----------+
Explanation: Joe is the only employee who earns more than his manager.
"""


import pandas as pd

def find_employees(employee: pd.DataFrame) -> pd.DataFrame:
    merged_df = employee.merge(employee, left_on="managerId", right_on="id", suffixes=("_x", "_y"))
    where_df = merged_df[merged_df["salary_x"] > merged_df["salary_y"]]
    return where_df[["name_x"]].rename(columns={"name_x": "Employee"})





# 182. Duplicate Emails
# https://leetcode.com/problems/duplicate-emails/description/
"""
Table: Person

+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| email       | varchar |
+-------------+---------+
id is the primary key (column with unique values) for this table.
Each row of this table contains an email. The emails will not contain uppercase letters.
 

Write a solution to report all the duplicate emails. Note that it's guaranteed that the email field is not NULL.

Return the result table in any order.

The result format is in the following example.

Example 1:

Input: 
Person table:
+----+---------+
| id | email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+
Output: 
+---------+
| Email   |
+---------+
| a@b.com |
+---------+
Explanation: a@b.com is repeated two times.
"""


type_table = """
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| email       | varchar |
+-------------+---------+
"""

data_table = """
+----+---------+
| id | email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+
"""


import pandas as pd

object_table = pd.DataFrame({'id': [1, 2, 3], 'email': ['a@b.com', 'c@d.com', 'a@b.com']})

def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
    # # return person.groupby('email').filter(lambda x: len(x) > 1)[['email']].drop_duplicates()
    # grouped = person.groupby('email').size() > 1
    # # return grouped
    # return person[person['email'].isin(grouped)]

    grouped = person.groupby("email")
    filtered = grouped.filter(lambda x: x["email"].count() > 1)
    return filtered[["email"]].drop_duplicates()
duplicate_emails(object_table)





# 183. Customers Who Never Order
# https://leetcode.com/problems/customers-who-never-order/description/
"""
Write a solution to find all customers who never order anything.

Return the result table in any order.

The result format is in the following example.

Example 1:

Input: 
Customers table:
+----+-------+
| id | name  |
+----+-------+
| 1  | Joe   |
| 2  | Henry |
| 3  | Sam   |
| 4  | Max   |
+----+-------+
Orders table:
+----+------------+
| id | customerId |
+----+------------+
| 1  | 3          |
| 2  | 1          |
+----+------------+
Output: 
+-----------+
| Customers |
+-----------+
| Henry     |
| Max       |
+-----------+
"""


import pandas as pd

def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    merged = customers.merge(orders, how="left", left_on="id", right_on="customerId", suffixes=("_x", "_y"))
    filtered = merged[merged["id_y"].isna()]
    return filtered[["name"]].rename(columns={"name": "Customers"})





# 196. Delete Duplicate Emails
# https://leetcode.com/problems/delete-duplicate-emails/
"""
Write a solution to delete all duplicate emails, keeping only one unique email with the smallest id.

For SQL users, please note that you are supposed to write a DELETE statement and not a SELECT one.

For Pandas users, please note that you are supposed to modify Person in place.

After running your script, the answer shown is the Person table. The driver will first compile and run your piece of code and then show the Person table. The final order of the Person table does not matter.

The result format is in the following example.

 

Example 1:

Input: 
Person table:
+----+------------------+
| id | email            |
+----+------------------+
| 1  | john@example.com |
| 2  | bob@example.com  |
| 3  | john@example.com |
+----+------------------+
Output: 
+----+------------------+
| id | email            |
+----+------------------+
| 1  | john@example.com |
| 2  | bob@example.com  |
+----+------------------+
Explanation: john@example.com is repeated two times. We keep the row with the smallest Id = 1.
"""


import pandas as pd

def delete_duplicate_emails(person: pd.DataFrame) -> None:
    person.sort_index(inplace=True, ascending=True)
    # person.sort_values(by="id", inplace=True, ascending=True)
    person.drop_duplicates(subset="email", keep="first", inplace=True)
    return person





# 197. Rising Temperature
# https://leetcode.com/problems/rising-temperature/description/
"""
Table: Weather

+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| recordDate    | date    |
| temperature   | int     |
+---------------+---------+
id is the column with unique values for this table.
There are no different rows with the same recordDate.
This table contains information about the temperature on a certain day.
 

Write a solution to find all dates' Id with higher temperatures compared to its previous dates (yesterday).

Return the result table in any order.

The result format is in the following example.

 

Example 1:

Input: 
Weather table:
+----+------------+-------------+
| id | recordDate | temperature |
+----+------------+-------------+
| 1  | 2015-01-01 | 10          |
| 2  | 2015-01-02 | 25          |
| 3  | 2015-01-03 | 20          |
| 4  | 2015-01-04 | 30          |
+----+------------+-------------+
Output: 
+----+
| id |
+----+
| 2  |
| 4  |
+----+
Explanation: 
In 2015-01-02, the temperature was higher than the previous day (10 -> 25).
In 2015-01-04, the temperature was higher than the previous day (20 -> 30).
"""


Weather = {
    "id": [1, 2, 3, 4],
    "recordDate": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
    "temperature": [10, 25, 20, 30]
}


import pandas as pd

def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:
    weather["nextDay"] = weather["recordDate"] + pd.Timedelta(days=1)
    merged = weather.merge(weather, how="left", left_on="recordDate", right_on="nextDay")
    where_temp_rising = merged[merged["temperature_x"] > merged["temperature_y"]]
    # return where_temp_rising["index"].rename(columns={"id_x": "Id"})
    return pd.DataFrame(where_temp_rising["recordDate_x"])
rising_temperature(object_table)






import pandas as pd
Weather = {
    "id": [1, 2, 3, 4],
    "recordDate": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
    "temperature": [10, 25, 20, 30]
}


def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:
    weather = pd.DataFrame(weather, index=weather["id"])
    weather["recordDate"] = pd.to_datetime(weather["recordDate"])
    weather["Date2"] = weather["recordDate"] + pd.Timedelta(days=1)
    merged = weather.merge(weather, left_on="recordDate", right_on="Date2")
    where = merged[merged["temperature_x"] > merged["temperature_y"]]
    return pd.DataFrame(where["id_x"]).rename(columns={"id_x": "Id"})
rising_temperature(Weather)







