import pandas as pd

## sql -> df : pd.read_sql()
## csv -> df : pd.read_csv(".csv")
## json -> df : pd.read_json(".json")
## df -> csv : df.to_csv(".csv", index=False)
## df -> sql : df.to_sql("<table_name>", sqlite_connection, if_exists="replace", index_label="index")
## df -> json : df.to_json(".json", indent=4, orient="records")
## dict -> df : pd.DataFrame(<dict>)
## df -> dict : df.to_dict()





## csv -> df
file = pd.read_csv("mtcars.csv")



## df -> csv
file.to_csv("mtcars.csv", index=False)
print(file.to_csv())



## dict -> df
emp_data = {
    'name': ['John', 'Alice', 'Bob', 'Emma'],
    'managerId': [1, 2, 2, 1],
    'salary': [50000, 60000, 55000, 48000]
}

empleyees = pd.DataFrame(emp_data)
# empleyees[empleyees["salary"] > 51000]

empleyees.to_csv("empleyees.csv", index=False)

## df –> dict
empleyees.to_dict()
empleyees.to_dict("records")





# Create an in-memory SQLite database.
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)
empleyees.to_sql(name='emplo_db_name', con=engine)

# Fetch data from in-memory database.
from sqlalchemy import text
with engine.connect() as conn:
    conn.execute(text("SELECT * FROM emplo_db_name")).fetchall()



## df -> db # Save database to SQLite3
from sqlalchemy import create_engine
engine = create_engine('sqlite:///employees.db', echo=True)
sqlite_connection = engine.connect()
empleyees.to_sql("employees", sqlite_connection, if_exists="replace", index_label="index")
sqlite_connection.close()

## db -> df
from sqlalchemy import create_engine
engine = create_engine('sqlite:///mtcars.db', echo=True)
with engine.connect() as conn, conn.begin():
    data_from_db = pd.read_sql("mtcars", conn, index_col="index")
data_from_db










# mtcars
import pandas as pd

mtcars = pd.read_csv('https://gist.githubusercontent.com/ZeccaLehn/4e06d2575eb9589dbe8c365d61cb056c/raw/64f1660f38ef523b2a1a13be77b002b98665cdfe/mtcars.csv')
mtcars.columns
mtcars.rename(columns={'Unnamed: 0':'brand'}, inplace=True)

## df -> csv
mtcars.to_csv("mtcars.csv", index=False)

## df -> db
from sqlalchemy import create_engine
engine = create_engine('sqlite:///mtcars.db')
sqlite_connection = engine.connect()
mtcars.to_sql("mtcars", sqlite_connection, if_exists="replace")
sqlite_connection.close()








## df -> json
mtcars.to_json("mtcars.json", indent=4, orient="records")

emp_data = {
    'name': ['John', 'Alice', 'Bob', 'Emma'],
    'managerId': [1, 2, 2, 1],
    'salary': [50000, 60000, 55000, 48000]
}
emp_df = pd.DataFrame(emp_data)
emp_df.to_json("employee.json", indent=4, orient="records")




## json -> df
mtcars_read = pd.read_json("mtcars.json")







