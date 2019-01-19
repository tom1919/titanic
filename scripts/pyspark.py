# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:01:01 2019

@author: tommy
"""


# =============================================================================
# The purpose of this script is to illustrate basic ML workflow with pyspark. 
# The titanic dataset from kaggle.com is used as an example for binary 
# classification. The target variable is 'Survived' and the goal is predict 
# whether or not a passenger survived the titanic ship crash.
# =============================================================================


#%%
# import libraries and create spark session 

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnull, count, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.ml.tuning as tune

spark = SparkSession.builder.appName('abc').getOrCreate()


#%%
# read in data and take a look

df = spark.read.csv('C:/Users/tommy/Google Drive/titanic/data/raw/train.csv', 
                    header=True, inferSchema=True)

df.show(15)

df.count()

df.filter(df['Survived'] == 1).count()
df.filter(df['Survived'] == 0).count()

df.groupBy('sex').agg({'Survived' : 'sum'}).show()

#%%
#  Column selection and make sex variable numeric 

# columns with NAs
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

# select target variable and the predictors that don't have NAs.
df = df.select('Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare')

# make sex variable 0 or 1
df = df.withColumn('Sex', when(col('Sex') == 'male', 0).otherwise(1) )

#%%
# Model data prep and split

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']

# Generate Features Vector 
va = VectorAssembler(inputCols=features, outputCol="features")
modelprep = va.transform(df).withColumn("label", col('Survived'))

# split 80-20 for train and test sets
train, test = modelprep.randomSplit([.8,.2])


#%%
# fit model using 5 fold cross validation. 

# Create a LogisticRegression Estimator
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label')

# Create a BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameters
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, np.arange(0, 1.1, 0.1))

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator,
                         numFolds = 5)

# Fit cross validation models
models = cv.fit(train)

#%%
# Evaluate model on test set and look at coefficients

# Extract the best model
best_lr = models.bestModel

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

# coefficients for independent variables
best_lr.coefficients
best_lr.intercept


