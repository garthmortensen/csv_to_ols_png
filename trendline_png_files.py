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

today = datetime.today().strftime('%Y%m%d')

# loop all csv
for file in os.listdir(data_in):
    if file.endswith(".csv"):

        df = pd.read_csv(os.path.join(data_in, file))
        x = df[["id"]]
        y = df[["value"]]

        regressor = LinearRegression()
        regressor.fit(x, y)
        y_pred = regressor.predict(x)
        
        plt.plot(x, y, color='red')
        plt.plot(x, regressor.predict(x), color='blue')
        plt.title(f"{file}: Rows per id w/trend")
        plt.xlabel('id')
        plt.ylabel('value')
        plt.show()
        plt.savefig(f"{data_out}/{file}_{today}.png")
