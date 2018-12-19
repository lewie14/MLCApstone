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

#*********************************************************************************
#Add a column sign_code to the df which maps the categories to numerical data
sign_mapping = {
					"aries": 0,
					"aries and it&rsquo;s fun to think about": 0,
					"aries but it doesn&rsquo;t matter": 0,
					"aries and it matters a lot": 0,
					"taurus": 1, 
					"taurus and it&rsquo;s fun to think about": 1,
					"taurus but it doesn&rsquo;t matter": 1,
					"taurus and it matters a lot": 1,
					"gemini": 2, 
					"gemini and it&rsquo;s fun to think about": 2,
					"gemini but it doesn&rsquo;t matter": 2,
					"gemini and it matters a lot": 2,
					"cancer": 3, 
					"cancer and it&rsquo;s fun to think about": 3,
					"cancer but it doesn&rsquo;t matter": 3,
					"cancer and it matters a lot": 3,
					"leo": 4, 
					"leo and it&rsquo;s fun to think about": 4,
					"leo but it doesn&rsquo;t matter": 4,
					"leo and it matters a lot": 4,
					"virgo": 5, 
					"virgo and it&rsquo;s fun to think about": 5,
					"virgo but it doesn&rsquo;t matter": 5,
					"virgo and it matters a lot": 5,
					"libra":6, 
					"libra and it&rsquo;s fun to think about": 6,
					"libra but it doesn&rsquo;t matter": 6,
					"libra and it matters a lot": 6,
					"scorpio":7, 
					"scorpio and it&rsquo;s fun to think about": 7,
					"scorpio but it doesn&rsquo;t matter": 7,
					"scorpio and it matters a lot": 7, 
					"sagittarius": 8,
					"sagittarius and it&rsquo;s fun to think about": 8,
					"sagittarius but it doesn&rsquo;t matter": 8,
					"sagittarius and it matters a lot": 8,
					"capricorn":9, 
					"capricorn and it&rsquo;s fun to think about": 9,
					"capricorn but it doesn&rsquo;t matter": 9,
					"capricorn and it matters a lot": 9,
					"aquarius":10, 
					"aquarius and it&rsquo;s fun to think about": 10,
					"aquarius but it doesn&rsquo;t matter": 10,
					"aquarius and it matters a lot": 10,
					"pisces":11,
					"pisces and it&rsquo;s fun to think about": 11,
					"pisces but it doesn&rsquo;t matter": 11,
					"pisces and it matters a lot": 11,
					"unanswered":12
					}
df["sign_code"] = df.sign.map(sign_mapping)
#Set Nans to unanswered i.e. 12
df["sign_code"].fillna(value=12, inplace=True)
print(df.sign_code.value_counts())
#********************************************************************************************************

plt.scatter(df[['sign_code']], df[['height']], alpha=0.4)
plt.show()

