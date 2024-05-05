# from convert.create_db import get_db
from create_db import get_db
# from convert import create_db



table_name = "Activite"

ascii_type_table = """
+--------------+---------+
| Column Name  | Type    |
+--------------+---------+
| player_id    | int     |
| device_id    | int     |
| event_date   | date    |
| games_played | int     |
+--------------+---------+
"""

ascii_object_table = """
+-----------+-----------+------------+--------------+
| player_id | device_id | event_date | games_played |
+-----------+-----------+------------+--------------+
| 1         | 2         | 2016-03-01 | 5            |
| 1         | 2         | 2016-05-02 | 6            |
| 2         | 3         | 2017-06-25 | 1            |
| 3         | 1         | 2016-03-02 | 0            |
| 3         | 4         | 2018-07-03 | 5            |
+-----------+-----------+------------+--------------+
"""


object_table = get_db(ascii_object_table, ascii_type_table)
# object_table = create_db.get_db(ascii_object_table, ascii_type_table)
object_table.to_dict("list")


# from convert_pd.py
# Create an in-memory SQLite database.
# from sqlalchemy import create_engine
# engine = create_engine('sqlite://', echo=False)
# object_table.to_sql(name=table_name, con=engine)

## df -> db # Save database to SQLite3
from sqlalchemy import create_engine
engine = create_engine('sqlite:///leetcode.sqlite3', echo=False)
sqlite_connection = engine.connect()
object_table.to_sql(name=table_name, con=sqlite_connection, if_exists="replace", index_label="index")


# Fetch data from in-memory database.
from sqlalchemy import text
with engine.connect() as conn:
    lines = conn.execute(text(
        "SELECT w2.'index'\
        FROM Weather w1\
        INNER JOIN Weather w2 ON DATE(w2.recordDate) = DATE(w1.recordDate, '+1 day')\
        WHERE w2.temperature > w1.temperature\
        "
        )).fetchall()
    for line in lines:
        print(line)





orders = object_table
# index, ind, pk column name
object_table.index.name

import pandas as pd
def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    earliest_dates = activity.groupby("player_id")["event_date"].min()
    # filtered = pd.merge(activity, earliest_dates, how='inner', on=["player_id", "event_date"])
    filtered = pd.merge(activity, earliest_dates, how='inner', left_on="player_id", right_on="event_date")
    return filtered

    return filtered[["player_id", "event_date"]]
game_analysis(object_table)



