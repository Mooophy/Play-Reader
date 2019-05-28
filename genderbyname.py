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

baby= pd.read_csv("data/name.csv")
baby.columns=['name','sex']
baby=baby.drop_duplicates()
baby.sex = pd.to_numeric(baby.sex, errors='coerce')
baby.head()
df_X=features(baby.name)
df_y = baby.sex
dfX_train,dfX_test,dfy_train,dfy_test = train_test_split(df_X,df_y,test_size = 0.2)
dv = DictVectorizer()
dv.fit_transform(dfX_train)
dclf = MultinomialNB()
my_xfeatures = dv.transform(dfX_train)
dclf.fit(my_xfeatures,dfy_train)
mnames=["king","sir","lord","prince"]
fnames=["queen","lady","princess","nurse"]

def genderclassify(name):
    for mname in mnames:
        if mname in name:
            return 0
    for fname in fnames:
        if fname in name:
            return 1
    vector = dv.transform(features([name])).toarray()
    return dclf.predict(vector)[0]
    
