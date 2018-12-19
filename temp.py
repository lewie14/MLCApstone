import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("profiles.csv")

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

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
#Check no nans left
print(df.sign_code.unique())
#********************************************************************************************************


#*********************************************************************************
#Add a column age_code to the df which maps the categories to numerical data
df['age_split'] = pd.cut(df['age'], [0, 18, 23, 29, 39, 49, 59, 69, 120], labels=['0-18', '19-23', '24-29', '30-39', '40-49', '50-59', '60-69', 'Above 70'])

print(df.age_split.value_counts())

age_split_map = {
    "0-18": 0,
    "19-23": 1,
    "24-29": 2,
    "30-39": 3,    
    "40-49": 4,
    "50-59": 5,
    "60-69": 6,
    "Above 70": 7
}

df["age_split_code"] = df.age_split.map(age_split_map)
#Set Nans to 24-29 i.e. 2
df["age_split_code"].fillna(value=2, inplace=True)
print(df.age_split_code.value_counts())
#Check no nans left
print(df.age_split_code.unique())
#*********************************************************************************

#********************************************************************************

#Set Nans in Height to ave height = 68 inches
df["height"].fillna(value= 68, inplace=True)
print(df.height.value_counts())
#Check no nans left
print(df.height.unique())
#********************************************************************************


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#find the length of each profile entry
df["essay_len"] = all_essays.apply(lambda x: len(x))
print(df.essay_len.head())

#Find how many of the word 'me' appears in a profile
df["me_count"] = all_essays.apply(lambda row: 0 if pd.isnull(row) else row.casefold().count("me"))
print(df.me_count.head())

#Find how many of the word 'love' appears in a profile
df["love_count"] = all_essays.apply(lambda row: 0 if pd.isnull(row) else row.casefold().count("love"))
print(df.love_count.head())

#Find how many of the word 'laughing' appears in a profile
df["laughing_count"] = all_essays.apply(lambda row: 0 if pd.isnull(row) else row.casefold().count("laughing"))
print(df.laughing_count.head())

#Find how many of the word 'work' appears in a profile
df["work_count"] = all_essays.apply(lambda row: 0 if pd.isnull(row) else row.casefold().count("work"))
print(df.work_count.head())

#Normalise

scaler = MinMaxScaler()

#Reshape the current tables into a column with many rows and scale the data between 0 and 1 for normalisation using max min scaling
df["sign_code"] = scaler.fit_transform(np.reshape(df[["sign_code"]], (-1,1)))
df["age_split_code"] = scaler.fit_transform(np.reshape(df[["age_split_code"]], (-1,1)))
df["essay_len"] = scaler.fit_transform(np.reshape(df[["essay_len"]], (-1,1)))
#df["essay_len"] = scaler.fit_transform(np.reshape(df[["essay_len"]], (-1,1)))
df["me_count"] = scaler.fit_transform(np.reshape(df[["me_count"]], (-1,1)))
df["love_count"] = scaler.fit_transform(np.reshape(df[["love_count"]], (-1,1)))
df["laughing_count"] = scaler.fit_transform(np.reshape(df[["laughing_count"]], (-1,1)))
df["work_count"] = scaler.fit_transform(np.reshape(df[["work_count"]], (-1,1)))




'''
plt.hist(df.age, range = (df.age.min(), df.age.max()))
plt.hist(df.height, range = (df.height.min(), df.height.max()))

plt.show()
'''
