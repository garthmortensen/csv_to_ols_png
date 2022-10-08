# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 07:42:35 2022

@author: garth

Read all input files, do an OLS, export as timestamped png
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

filepath = os.path.dirname(__file__)
data_in = os.path.join(filepath, "data_in")
data_out = os.path.join(filepath, "data_out")

today = datetime.today().strftime("%Y%m%d")

# loop all csv
df_list = []
for file in os.listdir(data_in):
    
    filename, file_extension = os.path.splitext(file)
    
    if file.endswith(".csv"):

        df = pd.read_csv(os.path.join(data_in, file))
        x = df[["insert_id"]]
        y = df[["rows"]]

        # cumsum shows total growth over time
        df["cumsum_rows"] = df["rows"].cumsum()
        y_sum = df[["cumsum_rows"]]

        # trendline. Could use average as alternative
        regressor = LinearRegression()
        regressor.fit(x, y)
        y_pred = regressor.predict(x)

        # plot dataframe        
        plt.plot(x, y, color="blue", label="rows")
        plt.plot(x, y_sum, color="green", label="cumulative sum rows")
        plt.plot(x, regressor.predict(x), color="red", label="regressed rows")
        plt.title(f"{file}, as of {today}: Rows per insert_id w/trend")
        plt.legend(loc="center right")
        plt.xlabel("insert_id")
        plt.ylabel("rows")
        plt.savefig(f"{data_out}/{filename}_{today}.png")
        plt.show()

    df_list.append(df)

# join all tables together into single plot
# this shows entire database growth
df_all = pd.concat(df_list)
df_all.drop("insert_id", axis=1, inplace=True)
df_all.sort_values("key", inplace=True)

# cumsum shows total growth over time
x = df_all[["key"]]
y = df_all[["rows"]]

df_all["cumsum_rows"] = df_all["rows"].cumsum()
y_sum = df_all[["cumsum_rows"]]

# plot dataframe
plt.plot(x, y, color="blue", label="rows")
plt.plot(x, y_sum, color="green", label="cumulative sum rows")
plt.title(f"{file}, as of {today}: Rows per insert_id w/trend")
plt.title("all_tables: Rows per insert_id w/trend")
plt.legend(loc="center right")
plt.xlabel("insert_id")
plt.ylabel("rows")
plt.savefig(f"{data_out}/unioned_{today}.png")
plt.show()


