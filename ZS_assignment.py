#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler


# In[2]:


train = pd.read_csv("train.csv")
test=pd.read_csv('test.csv')


# In[3]:


train.info()


# In[4]:


L=[
    "animal_id_outcome",
   "animal_type",
   "breed",
   "color",
   "intake_condition",
   "intake_type",
   "time_in_shelter_days",
   "sex_upon_outcome",
    "age_upon_outcome_(days)",
   "outcome_type",
    "train_or_test" ]


# In[5]:


train_mid = train.copy() 
train_mid['train_or_test'] = 'train'

test_mid = test.copy()
test_mid['train_or_test'] = 'test'

test_mid['outcome_type'] = 9 
alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True)


# In[6]:


alldata.drop(alldata.columns.difference(L),1,inplace=True)


# In[7]:


alldata


# In[8]:


print(len(alldata['breed'].unique()))
alldata['breed']=[row.split('/', 1)[0] for row in alldata['breed']]
print(len(alldata['breed'].unique()))
alldata['breed'] = alldata['breed'].str.replace('Mix', '')
print(len(alldata['breed'].unique()))


# In[9]:


print(len(alldata['color'].unique()))
alldata['breed']=[row.split('/', 1)[0] for row in alldata['color']]
print(len(alldata['breed'].unique()))


# In[10]:


breed = pd.get_dummies(alldata['breed'])
color = pd.get_dummies(alldata['color'])
# sexuponintake = pd.get_dummies(alldata['sex_upon_intake'])
sexuponOutcome = pd.get_dummies(alldata['sex_upon_outcome'])
animalType = pd.get_dummies(alldata['animal_type'])
intakecondition=pd.get_dummies(alldata['intake_condition'])
intaketype=pd.get_dummies(alldata['intake_type'])


# In[11]:


del alldata['color']
del alldata['breed']
# del alldata['sex_upon_intake']
del alldata['sex_upon_outcome']
del alldata['animal_type']
del alldata['intake_condition']
del alldata['intake_type']


# In[12]:


alldata['outcome_type'] = alldata['outcome_type'].replace("Adoption",0).replace("Died",1).replace("Disposal",2).replace("Euthanasia",3).replace("Missing",4).replace("Relocate",5).replace("Return to Owner",6).replace("Rto-Adopt",7).replace("Transfer",8)


# In[13]:


alldata = pd.concat([alldata, breed, color, sexuponOutcome, animalType, intakecondition, intaketype], axis=1)


# In[14]:


alldata


# In[15]:


train = alldata.query('train_or_test == "train"')
test = alldata.query('train_or_test == "test"')


# In[16]:


drop_col = ['animal_id_outcome', 'outcome_type','train_or_test']

train_feature = train.drop(columns=drop_col)
train_target = train['outcome_type']
test_feature = test.drop(columns=drop_col)

train_feature=StandardScaler().fit_transform(train_feature)
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0)


# In[17]:


test_feature=StandardScaler().fit_transform(test_feature)


# In[18]:


# Gradient Boosting Classifier==============

gradientboost = GradientBoostingClassifier(random_state=0)
gradientboost.fit(X_train, y_train)
print('='*20)
print('GradientBoostingClassifier')
print(f'accuracy of train set: {gradientboost.score(X_train, y_train)}')
print(f'accuracy of test set: {gradientboost.score(X_test, y_test)}')

gradientboost_prediction = gradientboost.predict(test_feature)
gradientboost_prediction


# In[19]:


test_predictions_dataset=test['animal_id_outcome'].to_frame()
test_predictions_dataset.insert(1,"outcome_type",gradientboost_prediction)
test_predictions_dataset["outcome_type"] = test_predictions_dataset["outcome_type"].replace(0,"Adoption").replace(1,"Died").replace(2,"Disposal").replace(3,"Euthanasia").replace(4,"Missing").replace(5,"Relocate").replace(6,"Return to Owner").replace(7,"Rto-Adopt").replace(8,"Transfer")
test_predictions_dataset.to_csv('submission.csv',index=False)


# In[ ]:




