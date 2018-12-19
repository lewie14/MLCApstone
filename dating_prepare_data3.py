import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#*******************************************************************************
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
#print(df.drinks_code.value_counts())
'''
gives a new column called drinks_code
1.0    47737
2.0     5957
0.0     3267
Name: drinks_code, dtype: int64

from original data
socially       41780
rarely          5957
often           5164
not at all      3267
very often       471
desperately      322
'''
df["drinks_code"].fillna(value=0, inplace=True)
#print(np.isfinite(df.drinks_code).all())
#print(np.isnan(df.drinks_code).any())
print(df.drinks_code.value_counts())
'''
Now
1.0    47737
0.0     6252 *************
2.0     5957
So I know the Nans have been reset to 0
'''
#print(df.drinks.unique())
#print(df.drinks_code.unique())
#print(df.drinks_code.head())
'''
['socially' 'often' 'not at all' 'rarely' nan 'very often' 'desperately'] ****Original
[1. 2. 0.] 														*************New column
0    1.0
1    2.0
2    1.0
3    1.0
4    1.0
'''

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
'''
gives a new column called smokes_code
0.0    49408
1.0     6827
2.0     3711
Name: smokes_code, dtype: int64

from original data
no                43896
sometimes          3787
when drinking      3040
yes                2231
trying to quit     1480
'''


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
'''
gives a new column called diet_code
0.0    54258
1.0     4986
2.0      702
Name: diet_code, dtype: int64

from original data
mostly anything        16585
anything                6183
strictly anything       5113
mostly vegetarian       3444
mostly other            1007
strictly vegetarian      875
vegetarian               667
strictly other           452
mostly vegan             338
other                    331
strictly vegan           228
vegan                    136
mostly kosher             86
mostly halal              48
strictly halal            18
strictly kosher           18
halal                     11
kosher                    11
'''


#*********************************************************************************
#Add a column body_type_code to the df which maps the categories to numerical data
#Healthy  : 0
#Average  : 1
#Unhealthy: 2

body_type_mapping = {
					"average": 1, 
					"fit": 0, 
					"athletic": 0, 
					"thin": 1, 
					"curvy": 1, 
					"a little extra": 1, 
					"skinny":2, 
					"full figured":1, 
					"overweight":2, 
					"jacked":0, 
					"used up":2, 
					"rather not say":2
					}
'''
body_type_mapping = {
					"average": 2, 
					"fit": 5, 
					"athletic": 4, 
					"thin": 3, 
					"curvy": 1, 
					"a little extra": 7, 
					"skinny":6, 
					"full figured":8, 
					"overweight":9, 
					"jacked":10, 
					"used up":11, 
					"rather not say":12
					}
'''
df["body_type_code"] = df.body_type.map(body_type_mapping)
#Set Nans to Average i.e. 1
df["body_type_code"].fillna(value=1, inplace=True)
print(df.body_type_code.value_counts())

'''
gives a new column called body_type_code
1.0    32221
0.0    24951
2.0     2774
Name: body_type_code, dtype: int64


from original data
average           14652
fit               12711
athletic          11819
thin               4711
curvy              3924
a little extra     2629
skinny             1777
full figured       1009
overweight          444
jacked              421
used up             355
rather not say      198
'''

#*********************************************************************************
#Add a column offspring_code to the df which maps the categories to numerical data
#NoKids  : 0
#HasKids  : 1


offspring_mapping = {	"doesn&rsquo;t have kids": 0, 
						"doesn&rsquo;t have kids, but might want them": 0, 
						"doesn&rsquo;t have kids, but wants them": 0, 
						"doesn&rsquo;t want kids": 0, 
						"has kids": 1, 
						"has a kid": 1, 
						"doesn&rsquo;t have kids, and doesn&rsquo;t want any":0, 
						"has kids, but doesn&rsquo;t want more":1, 
						"has a kid, but doesn&rsquo;t want more ":1, 
						"has a kid, and might want more":1, 
						"wants kids":0, 
						"might want kids":0,
						"has kids, and might want more":1,
						"has a kid, and wants more":1,
						"has kids, and wants more":1
						}
						
df["offspring_code"] = df.offspring.map(offspring_mapping)
#Set Nans to Average i.e. 0
df["offspring_code"].fillna(value=0, inplace=True)
print(df.offspring_code.value_counts())
#gives a new column called offspringcode
#0.0    55302
#1.0     4644
#Name: offspring_code, dtype: int64






'''
#Join all eassays together
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays as one string
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#Create new column 
df["essay_len"] = all_essays.apply(lambda x: len(x))
'''



'''
#Normalize or our body type data
feature_data = df[['smokes_code', 'drinks_code', 'diet_code']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()#******preprocessing' is not defined
x_scaled = min_max_scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
'''

#********************************MLR*****************************************

from sklearn.model_selection import train_test_split

#Create a DataFrame x that selects drinks_code, smokes_code, diet_code, offspring_code from the main df DataFrame:

x = df[['drinks_code', 'smokes_code', 'diet_code', 'offspring_code']]

y = df['body_type_code']

#Use scikit-learn's train_test_split() method to split x into 80% training set and 20% testing set and generate:

#x_train
#x_test
#y_train
#y_test
#Set the random_state to 6.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
gives
(47956, 4)
(11990, 4)
(47956,)
(11990,)
'''

from sklearn.linear_model import LinearRegression

#Create a Linear Regression model
mlr = LinearRegression()

#Fit the model to the x_train and y_train data
mlr.fit(x_train, y_train)

#Use the model to predict y-values from x_test. Store the predictions in a variable called y_predict
y_predict = mlr.predict(x_test)

#Test the model against data in row 30 of the csv which belongs to a 'fit'person so should bring back 'Healthy'
'''
print(df.body_type.head(28),df.body_type_code.head(28))
print("Hello")
print(df.body_type_code[27])
print(df.drinks_code[27])
print(df.smokes_code[27])
print(df.diet_code[27])
print(df.offspring_code[27])

#row_30 = [[df.drinks_code[29], df.smokes_code[29], df.diet_code[29], df.offspring_code[29]]]
'''
row_1 = [[1,0,0,0]]

predict = mlr.predict(row_1)
print("Predicted Health: %.0f" % predict)

plt.scatter(y_test, y_predict, alpha=0.4)

plt.xlabel("Body Type: $Y_i$")
plt.ylabel("Predicted Body Type: $\hat{Y}_i$")
plt.title("Actual Body Type vs Predicted Body Type")

plt.show()

