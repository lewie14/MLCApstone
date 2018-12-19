import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
#from svm_visualization import draw_boundary

#Create your df here:
df = pd.read_csv("profiles.csv")

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

#******************************************************************************
#Make a column of new signs answers which relate to the numbers*****************
#ie 0 : Aries
#  1 : Taurus etc

number_to_sign_mapping = {
						0 :	"Aries",
						1 : "Taurus",
						2 : "Gemini",
						3 : "Cancer",
						4 : "Leo",
						5 : "Virgo",
						6 : "Libra",
						7 : "Scorpio",
						8 : "Sagittarius",
						9 : "Capricorn",
						10 : "Aquarius",
						11 : "Pisces",
						12 : "Unanswered"
						}
							
df["actual_sign"] = df.sign_code.map(number_to_sign_mapping)
print("This is the actual sign array")
print(df.actual_sign)
print(df.actual_sign.value_counts())
#*****************************************************************************







#*********************************************************************************
#Add a column diet_code to the df which maps the categories to numerical data
#MeatEater : 0
#Vegetarian : 1
#Vegan: 2
diet_mapping = {
				"mostly anything": 0, 
				"anything": 0, 
				"strictly anything": 0, 
				"mostly vegetarian": 1, 
				"mostly other": 0, 
				"strictly vegetarian": 1, 
				"vegetarian":1, 
				"strictly other":0, 
				"mostly vegan":2, 
				"other":0, 
				"strictly vegan":2, 
				"vegan":2, 
				"mostly kosher":0, 
				"mostly halal":0, 
				"strictly halal":0, 
				"strictly kosher":0, 
				"halal":0, 
				"kosher":0
				}
				
df["diet_code"] = df.diet.map(diet_mapping)
#Set Nans to MeatEater i.e. 0
df["diet_code"].fillna(value=0, inplace=True)
print(df.diet_code.value_counts())
#************************************************************************

#***********************************************************************
#Add a column drinks_code to the df which maps the categories to numerical data
#No : 0
#Moderate : 1
#Yes: 2
drink_mapping = {
				"not at all": 0, 
				"rarely": 1, 
				"socially": 1, 
				"often": 2, 
				"very often": 2, 
				"desperately": 2
				}
				
df["drinks_code"] = df.drinks.map(drink_mapping)

#Change the Nans to 0 i.e not at all
df["drinks_code"].fillna(value=0, inplace=True)
print(df.drinks_code.value_counts())
#***********************************************************************


#********************************************************************************
#Add a column smokes_code to the df which maps the categories to numerical data
#No : 0
#Moderate : 1
#Yes: 2
smokes_mapping = {
				"no": 0, 
				"sometimes": 1, 
				"when drinking": 1, 
				"yes": 2, 
				"trying to quit": 2
				}
				
df["smokes_code"] = df.smokes.map(smokes_mapping)
#Set Nans to No i.e. 0
df["smokes_code"].fillna(value=0, inplace=True)
print(df.smokes_code.value_counts())
#***********************************************************************

#********************************************************************************
#Set Nans age to ave age 27
df["age"].fillna(value=27, inplace=True)
print(df.age.value_counts())
#Check no nans left
print(df.age.unique())
#********************************************************************************


#********************************************************************************

#Set Nans in Height to ave height = 68 inches
df["height"].fillna(value= 68, inplace=True)
print(df.height.value_counts())
#Check no nans left
print(df.height.unique())
#********************************************************************************


#*******************************************************************************
#Create an array of the essay columns
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

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



#*************************Normalise the data******************************

scaler = MinMaxScaler()

#Reshape the current tables into a column with many rows and scale the data between 0 and 1 for normalisation using max min scaling
df["sign_code"] = scaler.fit_transform(np.reshape(df[["sign_code"]], (-1,1)))
df["height"] = scaler.fit_transform(np.reshape(df[["height"]], (-1,1)))
#df["essay_len"] = scaler.fit_transform(np.reshape(df[["essay_len"]], (-1,1)))
df["me_count"] = scaler.fit_transform(np.reshape(df[["me_count"]], (-1,1)))
df["love_count"] = scaler.fit_transform(np.reshape(df[["love_count"]], (-1,1)))
df["laughing_count"] = scaler.fit_transform(np.reshape(df[["laughing_count"]], (-1,1)))
df["work_count"] = scaler.fit_transform(np.reshape(df[["work_count"]], (-1,1)))


#****************************Multiple Linear Regression**************************

#independent variables
x = df[['sign_code', 'height', 'me_count', 'love_count', 'laughing_count', 'work_count']]


#dependent variable
y = df[['age']]

#Create training and test sets
x_train, x_test, y_train, y_test = train_test_split(
													x, 
													y, 
													train_size = 0.8, 
													test_size = 0.2, 
													random_state=6
													)

#Check the size is correct
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

#Create multiple linear regression model
mlr = LinearRegression()

#Fit the model
# finds the coefficients and the intercept value
mlr.fit(x_train, y_train)

#Use the model to predict y-values from x_test. 
#Store the predictions in a variable called y_predict
y_predict = mlr.predict(x_test)

'''
#Code to plot Actual Age vs Predicted Age using all 6 features
plt.scatter(y_test, y_predict, alpha=0.4)
plt.plot(range(80), range(80))
plt.xlabel("Age: $Y_i$")
plt.ylabel("Predicted Age: $\hat{Y}_i$")
plt.title("Actual Age vs Predicted Age")
'''

#Code to plot individual features against Actual Age
#plt.scatter(df[['sign_code']], df[['age']], alpha=0.4)
#plt.scatter(df[['height']], df[['age']], alpha=0.4)
#plt.scatter(df[['me_count']], df[['age']], alpha=0.4)
#plt.scatter(df[['love_count']], df[['age']], alpha=0.4)
#plt.scatter(df[['laughing_count']], df[['age']], alpha=0.4)
#plt.scatter(df[['work_count']], df[['age']], alpha=0.4)

#***********Evalaute the model***************
#Print out the coefficients
print(mlr.coef_)


#find the mean squared error regression loss for the training set.
print("Train score:")
print(mlr.score(x_train, y_train))

#find the mean squared error regression loss for the testing set.
print("Test score:")
print(mlr.score(x_test, y_test))

residuals = y_predict - y_test

#plt.scatter(y_predict, residuals, alpha=0.4)
#plt.title('Residual Analysis')
#plt.show()

#***********************KNN Regression*************************

regressor = KNeighborsRegressor(n_neighbors = 19, weights = "distance")
regressor.fit(x_train, y_train)

print("KNN Regressor Score:")
print(regressor.score(x_train, y_train))

#**********************Accuracy, Precision, F1 ********************
'''
print("Accuracy Score:")
print(accuracy_score(y_test, y_predict))
print("Recall Score:")
print(recall_score(y_test, y_predict))
print("Precision Score:")
print(precision_score(y_test, y_predict))
print("F1 Score:")
print(f1_score(y_test, y_predict))
'''
#***********************KNN Classification*************************
'''
#normalize data
df["smokes_code"] = scaler.fit_transform(np.reshape(df[["smokes_code"]], (-1,1)))
df["drinks_code"] = scaler.fit_transform(np.reshape(df[["drinks_code"]], (-1,1)))
df["diet_code"] = scaler.fit_transform(np.reshape(df[["diet_code"]], (-1,1)))


#Sign data
x = df[['sign_code']]

#Sign labels
y = df[['actual_sign']]


#Create training and test sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
													x, 
													y, 
													train_size = 0.8, 
													test_size = 0.2, 
													random_state=6
													)
#Check the size is correct
#print(len(training_data))
#print(len(training_labels))

accuracies = []

for k in range(1, 101):
	#Create the classifier object
	classifier = KNeighborsClassifier(n_neighbors = 3)

	#Train the classifier with the training set and training labels
	#and change training labels from column vector to 1d array
	classifier.fit(training_data, training_labels.values.ravel())


	#add each score to accuracies
	accuracies.append(classifier.score(validation_data, validation_labels))

#Make x axis	
k_list = range(1, 101)
#Plot the graph of K
plt.plot(k_list, accuracies)
plt.show()

'''
#**********************SVM Classifier************************************

#normalize data
df["smokes_code"] = scaler.fit_transform(np.reshape(df[["smokes_code"]], (-1,1)))
df["drinks_code"] = scaler.fit_transform(np.reshape(df[["drinks_code"]], (-1,1)))
df["diet_code"] = scaler.fit_transform(np.reshape(df[["diet_code"]], (-1,1)))

plt.scatter( x = df.smokes_code,
			 y = df.drinks_code,
			 c = df.sign_code,
			 cmap = plt.cm.coolwarm,
			 alpha = 0.25
			 )




#Create training and test sets
training_set, validation_set = train_test_split(
												df,  
												random_state=1
												)

classifier = SVC(kernel = 'rbf', gamma = 100, C = 100)
#Train the AI
classifier.fit(
		training_set[['smokes_code', 'drinks_code']],
		training_set.sign_code
		)

#Find accuracy
score = classifier.score(
		validation_set[['smokes_code', 'drinks_code']],
		validation_set.sign_code
		)

print(score)
'''
plt.show()

'''





















