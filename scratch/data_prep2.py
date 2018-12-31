# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 17:12:30 2018

@author: tommy
"""

#%%
import pandas as pd

#%%
# Read in data and combine them

train = pd.read_csv('C:/Users/tommy/Google Drive/titanic/data/raw/train.csv')
test = pd.read_csv('C:/Users/tommy/Google Drive/titanic/data/raw/test.csv')

train['set'] = 'train'
test['set'] = 'test'

train_test = pd.concat([train, test], sort = False)

#%%
# Handle missing values

train_test.Embarked.fillna('C', inplace = True)

med_fare = train[(train.Pclass == 3) & (train.Embarked == 'S')].Fare.median()

train_test['Fare'].fillna(med_fare, inplace = True)

#%%
# Feature engineering

train_test['Deck'] = train_test.Cabin.str[0]
train_test.Deck.fillna('U', inplace = True)

train_test['FamSize'] = train_test.SibSp + train_test.Parch + 1

train_test['NameLength'] = train_test['Name'].apply(lambda x: len(x))

train_test['Title'] = train_test.Name.str.extract('([A-Za-z]+)\.')
rare_title = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 
              'Rev', 'Sir', 'Jonkheer', 'Dona']
train_test.Title = train_test.Title.replace(rare_title, 'Rare')
train_test.Title = train_test.Title.replace({'Mlle': 'Miss',
                                             'Ms': 'Miss',
                                             'Mme': 'Mrs'})

bins = [0, 13, 19, 25, 35, 50, 81]
bin_names = ['child', 'teen', 'young adult', 'adult', 'mid age', 'sr']
age_cats = pd.cut(train_test.Age, bins, labels = bin_names)
train_test['age_cats'] = age_cats.astype('object').fillna('U')



train_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 
                'Age'], axis = 1, inplace = True)

#%%
# One hot encode categorical variables

cat_var = ['Sex', 'Embarked', 'Deck', 'Title', 'age_cats']
dummy = pd.get_dummies(train_test[cat_var])
train_test2 = pd.concat([train_test, dummy], axis = 1).drop(cat_var, axis = 1)

train2 = train_test2.drop('Survived', axis = 1)
test2 = train_test2.Survived
#%%





#%%

train_cb = train_test[train_test.set == 'train'].drop('set', axis = 1)

x_train = train_cb.drop(['Survived'], axis = 1)
y_train = train_cb.Survived




from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import catboost as cb



cat_mod = cb.CatBoostClassifier(eval_metric='Accuracy',
                                iterations=100, random_seed=42)

#cat_mod.fit(x_xgb, y_xgb)

results = cross_val_score(cat_mod, x_xgb, y_xgb, 
                          cv=KFold(n_splits = 10, random_state = 8))

results.mean()

#cat_feat_indx = np.where(x_train.dtypes == np.object)[0]
#cat_mod.fit(x_train, y_train, cat_features = cat_feat_indx)

#results = cross_val_score(cat_mod, x_train, y_train, 
#                          cv=KFold(n_splits = 3, random_state = 8))







pred = cat_mod.predict(test_cb, prediction_type = 'Class')

pd.DataFrame({'PassengerID' : test.PassengerId, 
              'Survived' : pred.astype(int)}).to_csv(
'C:/Users/tommy/Google Drive/titanic/data/cat_boost.csv', index = False)



#%%


import xgboost as xgb

train_xgb = train_test2[train_test.set == 'train'].drop('set', axis = 1)

x_xgb = train_xgb.drop(['Survived'], axis = 1)
y_xgb = train_xgb.Survived

xgb_mod = xgb.XGBClassifier()


results2 = cross_val_score(xgb_mod, x_xgb, y_xgb, 
                          cv=KFold(n_splits = 3, random_state = 8))

results2.mean()






















