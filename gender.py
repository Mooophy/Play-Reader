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
        'last3-letter':name[-3:],

    }
features = np.vectorize(features)
baby= pd.read_csv(../data/"name.csv")
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
def genderclassify(df):
    predict=np.full(shape=len(df.name),fill_value=0,dtype=np.int)
    for k in range(len(df.name)):
    #    if predict[k]!=2:
            test_name = [df.name[k]]
            vector = dv.transform(features(test_name)).toarray()
            predict[k]=dclf.predict(vector)
    df['classified']=predict
    #
    for i in df[df.name.str.contains("King")].classified.index:
        df.classified[i]=0
    for i in df[df.name.str.contains("Sir")].classified.index:
        df.classified[i]=0
    for i in df[df.name.str.contains("Lord")].classified.index:
        df.classified[i]=0
    for i in df[df.name.str.contains("Prince")].classified.index:
        df.classified[i]=0
    for i in range(len(df.name)):
        if df.name.values[i][-2:] == 'er':#or
            df.classified[i]=0
        if df.name.values[i][-2:] == 'or':#or
            df.classified[i]=0
    for i in df[df.name.str.contains("Queen")].classified.index:
        df.classified[i]=1
    for i in df[df.name.str.contains("Lady")].classified.index:
        df.classified[i]=1
    for i in df[df.name.str.contains("Princess")].classified.index:
        df.classified[i]=1
    for i in df[df.name.str.contains("Nurse")].classified.index:
        df.classified[i]=1
