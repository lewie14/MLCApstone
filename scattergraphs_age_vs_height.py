import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Create your df here:
df = pd.read_csv("profiles.csv")

#********************************************************************************

#Set Nans in Height to ave height = 68 inches
df["height"].fillna(value= 68, inplace=True)
print(df.height.value_counts())

#********************************************************************************

#********************************************************************************

#Set Nans in age to ave age = 35 inches
df["age"].fillna(value= 35, inplace=True)
print(df.age.value_counts())

#********************************************************************************

plt.scatter(df[['age']], df[['height']], alpha=0.4)
plt.show()
