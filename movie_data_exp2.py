import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from patsy import dmatrices
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

movies=pd.read_csv('tmdb_5000_movies.csv', usecols=['genres','budget','id','revenue','title','release_date', 'vote_average', 'vote_count'])
mov=pd.read_csv('tmdb_5000_credits.csv')

# changing the genres column from json to string
movies['genres']=movies['genres'].apply(json.loads)
for index,i in zip(movies.index,movies['genres']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))# the key 'name' contains the name of the genre
    try:
        genre = list1[0]
    except:
        genre = 'Unspecified'
    movies.loc[index,'genres']=genre

mov['cast']=mov['cast'].apply(json.loads)
for index,i in zip(mov.index,mov['cast']):
    list1=[]
    for j in range(len(i)):
        list1.append((i[j]['name']))
    try:
        actor = list1[0]
    except:
        actor = 'Unspecified'
    mov.loc[index,'cast']=actor

mov['crew']=mov['crew'].apply(json.loads)

def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']

mov['crew']=mov['crew'].apply(director)
mov.rename(columns={'crew':'director'},inplace=True)



movies=movies.merge(mov,left_on='id',right_on='movie_id',how='left')

movies['profitability'] = movies['revenue'] - movies['budget']

y = movies.revenue.values
x = movies.vote_average.values
print(x.shape, y.shape)
length = 4083
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

#linear regression

regr = linear_model.LinearRegression()
regr.fit(x,y)

plt.scatter(x, y,  color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

# Note the difference in argument order
# lm = linear_model.LinearRegression()
# model = lm.fit(X,y)

# predictions = lm.predict(X)
# print(predictions)[0:5]

# model.summary()

print(movies.describe())

#neural network

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = np.array([movies.budget.values,movies.vote_average.values,movies.vote_count.values,movies.genres.values])
Y = np.array(movies.revenue.values - movies.budget.values)
X = X.transpose()
Y = (Y > 0) 

# encode strings as integers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 3] = labelencoder_X_1.fit_transform(X[:, 3])

# Partition training/testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)	


model = Sequential()
model.add(Dense(output_dim = 3, init = 'uniform', activation = 'tanh', input_dim = 4))
model.add(Dense(output_dim = 2, init = 'uniform', activation = 'tanh', input_dim = 3))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 2, nb_epoch = 20)

y_pred = model.predict(X_test)
y_pred = (y_pred > .5)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

