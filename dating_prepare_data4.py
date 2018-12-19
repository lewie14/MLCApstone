import pandas as pd
import numpy as np
from matplotlib import pyplot as plt




#Create your df here:
df = pd.read_csv("profiles.csv")


#*********************************************************************************
#Add a column body_type_code to the df which maps the categories to numerical data
body_type_mapping = {
					"average": 0,
					"fit": 1, 
					"athletic": 2, 
					"thin": 3, 
					"curvy": 4, 
					"a little extra": 5, 
					"skinny":6, 
					"full figured":7, 
					"overweight":8, 
					"jacked":9, 
					"used up":10, 
					"rather not say":11
					}
df["body_type_code"] = df.body_type.map(body_type_mapping)
#Set Nans to Average i.e. 0
df["body_type_code"].fillna(value=0, inplace=True)
print(df.body_type_code.value_counts())


#*********************************************************************************
#Add a column pets_code to the df which maps the categories to numerical data
pets_mapping = {
					"likes dogs and likes cats": 0,
					"likes dogs": 1, 
					"likes dogs and has cats": 2, 
					"has dogs": 3, 
					"has dogs and likes cats ": 4, 
					"likes dogs and dislikes cats": 5, 
					"has dogs and has cats":6, 
					"has cats":7, 
					"likes cats ":8, 
					"has dogs and dislikes cats":9, 
					"dislikes dogs and likes cats":10, 
					"dislikes dogs and dislikes cats":11,
					"dislikes cats":12, 
					"dislikes dogs and has cats":13, 
					"dislikes dogs":14,
					"unanswered":15
					}
df["pets_code"] = df.body_type.map(pets_mapping)
#Set Nans to Average i.e. 15
df["pets_code"].fillna(value=1, inplace=True)
print(df.pets_code.value_counts())


#*********************************************************************************
#Add a column signs_code to the df which maps the categories to numerical data
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
df["sign_code"] = df.body_type.map(sign_mapping)
#Set Nans to unanswered i.e. 12
df["sign_code"].fillna(value=12, inplace=True)
print(df.sign_code.value_counts())


#*********************************************************************************
#Add a column status_code to the df which maps the categories to numerical data
status_mapping = {
					"single": 0,
					"seeing someone": 1, 
					"available": 2, 
					"married": 3, 
					"unknown": 4, 
					}
df["status_code"] = df.body_type.map(status_mapping)
#Set Nans to unknown i.e. 4
df["status_code"].fillna(value=1, inplace=True)
print(df.status_code.value_counts())



#***********************************************************************************
#Join all essays together
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)

# Combining the essays as one string
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#Create new column 
df["essay_len"] = all_essays.apply(lambda x: len(x))



'''
#**************************************************************************************
#make sure our numerical data all has the same weight

from sklearn.feature_extraction.text import CountVectorizer

feature_data = df[['body_type_code', 'pets_code', 'signs_code', 'status_code', 'essay_len']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()#******preprocessing' is not defined
x_scaled = min_max_scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
'''



#**********************************************************************************
#****************Classification Using K Nearest Neighbors**************************
#**********************************************************************************

#Normalize our data

def min_max_normalize(lst):
	minimum = min(lst)
	maximum = max(lst)
	normalized = []
	
	for value in lst:
		normalized_num = (value - minimum) / (maximum - minimum)
		normalized.append(normalized_num)
		
	return normalized

min_max_normalize(df["body_type_code"])
#min_max_normalize(df["pets_code"])#***Division by zero error
#min_max_normalize(df["signs_code"])
#min_max_normalize(df["status_code"])
#min_max_normalize(df["essay_len"])


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5)

feature_data = df[['body_type', 'pets', 'sign', 'status']]
feature_labels = df[['body_type_code', 'pets_code', 'sign_code', 'status_code']]

training_points = feature_data
training_labels = feature_labels
classifier.fit(training_points, training_labels)


unknown_points = [
  [.45, .2, .5, .6, .3], 
  [.25, .8, .9, .7, .8],
  [.1, .1, .9, .2, .2]
]

guesses = classifier.predict(unknown_points)
print(guesses)




'''
df['age_bucket'] = pd.cut(df['age'], [0, 18, 23, 29, 39, 49, 59, 69, 120], labels=['0-18', '19-23', '24-29', '30-39', '40-49', '50-59', '60-69', 'Old'])

print(df.age_bucket.value_counts())

age_bucket_map = {
    "0-18": 0,
    "19-23": 1,
    "24-29": 2,
    "30-39": 3,    
    "40-49": 4,
    "50-59": 5,
    "60-69": 6,
    "Old": 7
}

df["age_bucket_code"] = df.age_bucket.map(age_bucket_map)
'''



















