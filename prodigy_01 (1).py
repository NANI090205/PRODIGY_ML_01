

import pandas as pd
import numpy as np
import csv

data=pd.read_csv('/content/Housing.csv')
data.head()

data.describe()

data.isnull().sum()

data[['area','bedrooms','bathrooms','price']].corr()

x=data[['area','bedrooms','bathrooms']].values.reshape(-1,3)
y=data[['price']].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=4294967295)
model.fit(x_train,y_train.reshape(-1,1))
print(model.intercept_, model.coef_)

y_pred=model.predict(x_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean absolute Error:", mean_absolute_error(y_test, y_pred))

def predict_house_price():
    print("\nProvide the house details to predict the price:")
    try:
        area = float(input("Enter the area in square feet: "))
        bedrooms = int(input("Enter the number of bedrooms: "))
        bathrooms = float(input("Enter the number of bathrooms: "))
        input_data = np.array([[area, bedrooms, bathrooms]])
        predicted_price = model.predict(input_data)[0]
        print(f"\nPredicted Price of the house:",predicted_price)

    except ValueError:
        print("Invalid input. Please enter valid numbers.")
predict_house_price()