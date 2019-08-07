# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:,4].values


# Encode the catogarical data of city "NewYork/California"

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
enc= OneHotEncoder(categorical_features=[3])
X = enc.fit_transform(X).toarray()

# Avoid dummy trap because we can have catorigal data in one coloumn
X=X[:,1:]

# Splitting the dataset into training set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test= train_test_split(X,Y, test_size=0.2, random_state =0 )

# Fitting MultipleLinear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predict test set results

