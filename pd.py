# pd.columns: colnames
# pd.shape: -> tuple (number of rows, number of columns); df.shape[0]
# pd.head(n): 
# pd.loc[pd.attr == x, ["attr", "attr2"]]; [which_rows, which_columns]
# pd[pd["id"] == 101]: WHERE id == 101
# pd[["<col_name>"]]: SELECT <col_name>
# pd["<col_name>"] = : add a new column
# pd.drop_duplicates(subset="<col_name>", keep="first", inplace=False)
# customers.email == customers["email"]
# pd.attr.notnull()
# pd.dropna(subset=["attr"])
# pd.rename(columns:{"old": "new})
# pd.attr.astype(int) == pd.astype({"attr": int})
# pd.attr.fillna(0): pd.attr = pd.attr.fillna(0)
# pd.concat([df1, df2])
# copy=True, inplace=False
# weather.pivot(index="month", columns="city", values="temperature")
# pd.sort_values(by="attr", ascending=False)


import pandas as pd

def createDataframe(student_data: list[list[int]]) -> pd.DataFrame:
    student = pd.DataFrame(student_data)
    return student.rename(columns={0: "student_id", 1: "age"})

def createDataframe(student_data: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame(student_data, columns=["student_id", "age"])


createDataframe([[1,15],[2,11],[3,11],[4,20]])




