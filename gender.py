import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

def features(name):
    return{
        'first-letter':name[0],
        'first2-letter':name[0:2],
        'first3-letter':name[0:3],
        'last-letter':name[-1],
        'last2-letter':name[-2:],
        'last3-letter':name[-3:]
    }

features = np.vectorize(features)
names= pd.read_csv("data/name.csv")
names.columns=['name','sex']
names=names.drop_duplicates()
names.sex = pd.to_numeric(names.sex, errors='coerce')
names.head()

x = features(names.name)
y = names.sex
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2)
dv = DictVectorizer()
dv.fit_transform(x_train)
mod = MultinomialNB()
mod.fit(dv.transform(x_train), y_train)

def gender_by_name(name):
    
    name = name.lower()
    ms = ["king","sir","lord","prince"]
    fs = ["queen","lady","princess","nurse", 'juliet']
    if(any(m in name for m in ms)): return 0
    if(any(f in name for f in fs)): return 1
    vector = dv.transform(features([name])).toarray()

    return mod.predict(vector)[0]
    
