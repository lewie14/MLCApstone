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
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)
print(df.drinks_code.value_counts())
'''
gives a new column called drinks_code
2.0    41780
1.0     5957
3.0     5164
0.0     3267
4.0      471
5.0      322
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
print(np.isfinite(df.drinks_code).all())
print(np.isnan(df.drinks_code).any())
print(df.drinks_code.value_counts())
'''
Now
2.0    41780
0.0     6252 *******
1.0     5957
3.0     5164
4.0      471
5.0      322
So I know the Nans have been reset to 0
'''
#print(df.drinks.unique())
#print(df.drinks_code.unique())
#['socially' 'often' 'not at all' 'rarely' 0 'very often' 'desperately']
#[2. 3. 0. 1. 4. 5.]


#********************************************************************************
#Add a column smokes_code to the df which maps the categories to numerical data
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)
print(df.smokes_code.value_counts())
'''
gives a new column called smokes_code
0.0    43896
1.0     3787
2.0     3040
3.0     2231
4.0     1480
Name: smokes_code, dtype: int64

from original data
no                43896
sometimes          3787
when drinking      3040
yes                2231
trying to quit     1480
'''
#Set Nans to No i.e. 0
df["smokes_code"].fillna(value=0, inplace=True)






#*********************************************************************************
#Add a column diet_code to the df which maps the categories to numerical data
diet_mapping = {"mostly anything": 0, "anything": 1, "strictly anything": 2, "mostly vegetarian": 3, "mostly other": 4, "strictly vegetarian": 5, "vegetarian":6, "strictly other":7, "mostly vegan":8, "other":9, "strictly vegan":10, "vegan":11, "mostly kosher":12, "mostly halal":13, "strictly halal":14, "strictly kosher":15, "halal":16, "kosher":17}
df["diet_code"] = df.diet.map(diet_mapping)
print(df.diet_code.value_counts())
'''
gives a new column called diet_code
0.0     16585
1.0      6183
2.0      5113
3.0      3444
4.0      1007
5.0       875
6.0       667
7.0       452
8.0       338
9.0       331
10.0      228
11.0      136
12.0       86
13.0       48
15.0       18
14.0       18
16.0       11
17.0       11
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

#Set Nans to Anything i.e. 1
df["diet_code"].fillna(value=1, inplace=True)


#*********************************************************************************
#Add a column body_type to the df which maps the categories to numerical data
body_type_mapping = {"average": 0, "fit": 1, "athletic": 2, "thin": 3, "curvy": 4, "a little extra": 5, "skinny":6, "full figured":7, "overweight":8, "jacked":9, "used up":10, "rather not say":11}
df["body_type_code"] = df.body_type.map(body_type_mapping)
print(df.body_type_code.value_counts())


'''
gives a new column called body_type_code

0.0     14652
1.0     12711
2.0     11819
3.0      4711
4.0      3924
5.0      2629
6.0      1777
7.0      1009
8.0       444
9.0       421
10.0      355
11.0      198
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

#Set Nans to Average i.e. 0
df["body_type_code"].fillna(value=0, inplace=True)

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

#**************Body_type and Drinks**********************
