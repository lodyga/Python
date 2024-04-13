# 175. Combine Two Tables
# https://leetcode.com/problems/combine-two-tables/
"""
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
    merged_df = employee.merge(employee, left_on="managerId", right_on="id", suffixes=("_emp", "_mgn"))
    filtered_df = merged_df[merged_df["salary_emp"] > merged_df["salary_mgn"]]
    result_df = filtered_df[["name_emp"]]
    result = result_df.rename(columns={"name_emp": "Employee"})
    # result_df.columns = ['Employee']
    # result = result_df[["name_emp"]].rename(columns={"name_emp": "Employee"})
    return result





# 182. Duplicate Emails
# https://leetcode.com/problems/duplicate-emails/description/
"""
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


import pandas as pd

per = pd.DataFrame({
    "id": [1, 2, 3], 
    "email": ["a@b.com", "b@c.com", "a@b.com"]
})

def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
    # # return person.groupby('email').filter(lambda x: len(x) > 1)[['email']].drop_duplicates()
    # grouped = person.groupby('email').size() > 1
    # # return grouped
    # return person[person['email'].isin(grouped)]

    grouped = person.groupby("email")
    filtered = grouped.filter(lambda x: x["email"].count() > 1)
    return filtered[["email"]].drop_duplicates()
duplicate_emails(per)





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
    joined = customers.merge(orders, how="left", left_on="id", right_on="customerId", suffixes=("_c", "_o"))
    filtered = joined[joined["customerId"].isna()]
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
    person.sort_values(by="id", inplace=True)
    person.drop_duplicates(subset="email", keep="first", inplace=True)





# 197. Rising Temperature
# https://leetcode.com/problems/rising-temperature/description/
"""
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
    weather = pd.DataFrame(weather, index=weather["id"])
    weather["Date2"] = weather["recordDate"]
    return weather
rising_temperature(Weather)






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


def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:

    # Create a new DataFrame shifted by one day
    weather_df_shifted = weather.copy()
    weather_df_shifted['recordDate'] += pd.Timedelta(days=1)
    weather_df_shifted.rename(
        columns={'temperature': 'next_day_temperature'}, inplace=True)

    # Merge both DataFrames on recordDate
    merged_df = pd.merge(weather, weather_df_shifted,
                         how='inner', on='recordDate', suffixes=("_c", "_o"))

    # Filter rows where temperature on next day is higher
    filtered_df = merged_df[merged_df['next_day_temperature']
                            < merged_df['temperature']]

    # Select only the 'id' column from the original DataFrame
    result = filtered_df
    return filtered_df[["id_c"]].rename(columns={"id_c": "Id"})





-- 

