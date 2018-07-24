#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 15:14:58 2018

@author: surbhikhandelwal
"""

# Import libararies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import CategoricalEncoder

#Load training & test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Store saleprice in a variable
saleprice_train = train.SalePrice

#concat test & training data so that operations on both dataset will be same
data = pd.concat([train.drop(['SalePrice'], axis =1) , test])

data.info()

categorical_cols = [x for x in data.columns if data[x].dtype == 'object']
int_cols = [x for x in data.columns if data[x].dtype == 'int64']
float_cols =[x for x in data.columns if data[x].dtype == 'float64']

#MSSubClass looks to be categorical so adding it in categorical_cols
categorical_cols.append('MSSubClass')
#Remove id and MsSubClass cols from int_cols
remove_int = ['MSSubClass', 'Id']
int_cols = [x for x in int_cols if x not in remove_int]
#concat from int_cols & float_cols
num_cols = int_cols + float_cols

data.fillna(0, inplace=True)


def encode_cat(dat):   
    """ functon to return a labeled data frame with one hot encoding """
    cat_encoder = CategoricalEncoder(encoding="onehot-dense")
    dat = dat.astype('str')
    dat_reshaped = dat.values.reshape(-1, 1)
    dat_1hot = cat_encoder.fit_transform(dat_reshaped)
    col_names = [dat.name + "_" + str(x) for x in list(cat_encoder.categories_[0])]
    return pd.DataFrame(dat_1hot, columns=col_names)
cat_df = pd.DataFrame()

for x in categorical_cols:
    cat_df = pd.concat([cat_df, encode_cat(data[x])], axis=1)
    
cat_df.index = data.index

final_data = pd.concat([data[num_cols], cat_df], axis =1)

data_train = final_data.iloc[:1460]
data_test = final_data.iloc[1460:]

X = data_train.values
test_final = data_test.values
y = saleprice_train.values

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
classifier = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(test_final)


test['SalePrice'] = y_pred
test[['Id', 'SalePrice']].to_csv('predictions/xgboost_tree_2.csv', index=False)

#Rnadom Forest Regressor

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

#Applying K Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X =X, y = y, cv = 10)
accuracies.mean()

# Predicting a new result
y_pred = regressor.predict(test_final)

test['SalePrice'] = y_pred
test[['Id', 'SalePrice']].to_csv('predictions/randomforestreg.csv', index=False)




