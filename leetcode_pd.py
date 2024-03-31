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
    merged_df = employee.merge(employee, left_on='managerId', right_on='id', suffixes=('_emp', '_mgn'))
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

def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:
    return person.groupby('email').filter(lambda x: len(x) > 1)[['email']].drop_duplicates()



