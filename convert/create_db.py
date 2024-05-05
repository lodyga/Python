"""
Create db from ASCII-like tables. 

Table: Person

+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| personId    | int     |
| lastName    | varchar |
| firstName   | varchar |
+-------------+---------+

Table: Address

+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| addressId   | int     |
| personId    | int     |
| city        | varchar |
| state       | varchar |
+-------------+---------+


Person =

| personId | lastName | firstName |
| -------- | -------- | --------- |
| 1        | Wang     | Allen     |
| 2        | Alice    | Bob       |

Address =

| addressId | personId | city          | state      |
| --------- | -------- | ------------- | ---------- |
| 1         | 2        | New York City | New York   |
| 2         | 3        | Leetcode      | California |
"""


import pandas as pd


# import ASCII tables
ascii_object_table = """
| personId | lastName | firstName |
| -------- | -------- | --------- |
| 1        | Wang     | Allen     |
| 2        | Alice    | Bob       |
"""

ascii_type_table = """
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| personId    | int     |
| lastName    | varchar |
| firstName   | varchar |
+-------------+---------+
"""


# Convert ASCII-like table to list of dicts
def ascii_table_to_dict(ascii_object_table):
    lines = ascii_object_table.strip().split('\n')
    if lines[0][0] == "|":
        first_line = 0
    else:
        first_line = 1

    headers = [header.strip()
               for header in lines[first_line].split('|') if header.strip()]
    data = []

    for line in lines[first_line + 2:]:
        if line[0] == "|":
            values = [value.strip()
                      for value in line.split('|') if value.strip()]
            row = dict(zip(headers, values))
            # row = {headers[i]: int(values[i]) if values[i].isdigit() else values[i] for i in range(len(headers))}

            data.append(row)

    return data

object_table = ascii_table_to_dict(ascii_object_table)
"""
object_table = [
    {'personId': '1', 'lastName': 'Wang', 'firstName': 'Allen'},
    {'personId': '2', 'lastName': 'Alice', 'firstName': 'Bob'}
]

alt
"""
test_object_table = {
    'personId': ["1", "2"],
    "lastName": ["Wang", "Alice"],
    "firstName": ["Allen", "Bob"]
}

test_object_df = pd.DataFrame(test_object_table)
test_object_df.columns[0]
test_object_df = test_object_df.set_index(test_object_df.columns[0])




# Convert table to df
object_df = pd.DataFrame(object_table)
"""
object_df =
  personId lastName firstName
0        1     Wang     Allen
1        2    Alice       Bob
"""

# Convert df to dict
object_df.to_dict("list")
"""
{'personId': ['1', '2'], 'lastName': ['Wang', 'Alice'], 'firstName': ['Allen', 'Bob']}
{'personId': {0: '1', 1: '2'}, 'lastName': {0: 'Wang', 1: 'Alice'}, 'firstName': {0: 'Allen', 1: 'Bob'}}
"""


# Convert ASCII-like type table to dict
def ascii_type_to_dict(ascii_type_table):
    type_convert = {"int": "int", "varchar": "str", "date": "datetime64[ns]"}
    lines = ascii_type_table.strip().split("\n")
    type_dict = {}
    for line in lines:
        if "|" in line:
            values = [value.strip()
                      for value in line.split("|") if value.strip()]
            if values[0] != "Column Name":
                type_dict[values[0]] = type_convert[values[1]]
    return type_dict

type_table = ascii_type_to_dict(ascii_type_table)
"""
{'personId': 'int', 'lastName': 'str', 'firstName': 'str'}
"""


# apply dtypes to an object_table
def get_db(ascii_object_table, ascii_type_table):
    object_table = pd.DataFrame(ascii_table_to_dict(ascii_object_table))
    type_table = ascii_type_to_dict(ascii_type_table)

    for column, dtype in type_table.items():
        object_table[column] = object_table[column].astype(dtype)

    # object_table = object_table.set_index(object_table.columns[0]) # id and ind should be separate columns
    return object_table

result_df = get_db(ascii_object_table, ascii_type_table) 
"""
         lastName firstName
personId                   
1            Wang     Allen
2           Alice  

   personId lastName firstName
0         1     Wang     Allen
1         2    Alice       Bob
"""


"""
get_db(pd.DataFrame(ascii_table_to_dict(ascii_object_table)),
             ascii_type_to_dict(ascii_type_table)).dtypes
personId      int64
lastName     object
firstName    object
dtype: object
"""


result_db = get_db(ascii_object_table, ascii_type_table)

# if __name__ == "__main__":
#     result_df
