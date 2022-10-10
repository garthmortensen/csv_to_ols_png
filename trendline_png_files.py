# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 07:42:35 2022

@author: garth

Read all input files, do an OLS, a cumsum, export as timestamped png
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


def define_x_y(df, column1, column2):
    """ carve out x and y """
    x = df[[column1]]
    y = df[[column2]]

    return x, y


def add_cumsum(df):
    """ cumsum shows total growth over time """
    df["cumsum_rows"] = df["rows"].cumsum()
    y_sum = df[["cumsum_rows"]]

    return df, y_sum


def plot_and_save(x, y, y_sum, title, export_filename):
    """ produce, write and show plots. Show clears, so write before clear """
    plt.plot(x, y, color="blue", label="rows")
    plt.plot(x, y_sum, color="green", label="cumulative sum rows")
    plt.title(title)
    plt.legend(loc="center right")
    plt.xlabel("insert_id")
    plt.ylabel("rows")
    plt.savefig(export_filename)
    plt.show()


def main():
    """ do the work """
    df_list = []

    for file in os.listdir(data_in):

        filename, file_extension = os.path.splitext(file)

        if file.endswith(".csv"):

            df = pd.read_csv(os.path.join(data_in, file))

            x, y = define_x_y(df,
                              "insert_id",
                              "rows")
            df, y_sum = add_cumsum(df)

            # trendline. Could use average as alternative
            regressor = LinearRegression()
            regressor.fit(x, y)

            # plot dataframe
            plt.plot(x, regressor.predict(x), color="red", label="regressed rows")
            plot_and_save(x,
                          y,
                          y_sum,
                          f"{file}, as of {today}: Rows per insert_id w/trend",
                          f"{data_out}/{filename}_{today}.png")

            df_list.append(df)

    # join all tables together into single plot
    # this shows entire database growth
    df_all = pd.concat(df_list)
    df_all.drop("insert_id", axis=1, inplace=True)
    df_all.sort_values("key", inplace=True)

    # cumsum shows total growth over time
    x, y = define_x_y(df_all,
                      "key",
                      "rows")
    df_all, y_sum = add_cumsum(df_all)

    # plot dataframe
    plot_and_save(x,
                  y,
                  y_sum,
                  "all_tables: Rows per insert_id w/trend",
                  f"{data_out}/unioned_{today}.png")


main()
