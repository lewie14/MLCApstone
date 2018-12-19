import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

'''
#creates a pie chart for Body Type WORKS
df['body_type'].value_counts().plot(kind='pie', fontsize = 8, shadow = True, rotatelabels = True, pctdistance = 0.9, autopct = '%1.1f%%')
plt.axis('equal')
plt.ylabel('')
plt.title('Body Type', fontsize=20)
plt.show()
'''


'''
#creates a pie chart for Smokes - WORKS
df['smokes'].value_counts().plot(kind='pie', startangle = 180, shadow = True, autopct = '%1.1f%%')
plt.axis('equal')
plt.ylabel('')
plt.title('Smoking', fontsize=20)
#plt.legend(loc="best")
plt.show()
'''

'''
#histogram of the `age` columinksn: Note bins is how you want the data ranged, here every 20 WORKS CODECADEMY EXAMPLE
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()
'''

'''
#Bar Chart for Diet WORKS
df['diet'].value_counts().plot(kind='barh', fontsize = 8)
plt.title('Diet', fontsize=20)
plt.xlabel("Number of people")
plt.ylabel("Type")
plt.show()
'''

'''
#Bar Chart for Drinks WORKS
df['drinks'].value_counts().plot(kind='barh', fontsize = 8)
plt.title('Drinking Habits', fontsize=20)
plt.xlabel("Number of people")
plt.ylabel("Type")
plt.show()
'''

'''
#Hist Chart for Height WORKS
plt.hist(df.height, range = (df.height.min(), df.height.max()))
plt.xlabel("Height")
plt.ylabel("Number of people")
plt.show()
'''

#barchart for income
plt.bar(df.income.unique(), df.income.value_counts())
plt.show()
