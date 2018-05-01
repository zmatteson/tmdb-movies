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
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import cross_val_predict
import scikitplot as skplt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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


# for index in movies:
#     mov.loc[index,'release_date'] = int(mov.loc[index,'release_date'][:4])

movies.fillna(value=0,axis=1,inplace=True)

features = ['vote_average','budget','vote_count']

target = ['revenue']

train, test = train_test_split(movies,test_size=0.30)
train.head()


X_train = train[features].dropna()
y_train = train[target].dropna()
X_test = test[features].dropna()
y_test = test[target].dropna()


lin = linear_model.LinearRegression()
# train the model on the training set
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Real revenue")
plt.ylabel("Predicted revenue")
plt.show()

lin_score_train = lin.score(X_test, y_test)
lin_score_test = lin.score(X_train, y_train)
print("Training score: ",lin_score_train)
print("Testing score: ",lin_score_test)

# y = movies.revenue.values

# length = 4083
# y = y.reshape(-1, 1)

# x = preprocessing.scale(x)
# y = preprocessing.scale(y)

# regr = linear_model.LinearRegression()
# regr.fit(x,y)



# regr.summary()

# print(movies.describe())



