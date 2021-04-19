import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt


data = pd.read_csv("VikingsSeason.csv", sep=",")#Import CSV file containing the data

data = data[["Year","W","L","T","Div. Finish"]]#Select which data we will use to form our model

predict = "W"#Select our variable which we will try to predict

x = np.array(data.drop([predict], 1))#Create an array of all values except our value we will be predicting
y = np.array(data[predict])#Create an array of our predicted value
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
#Create our varibles. The train variables are what the computer will learn off of and the test values are
#the values that will be tested against.


best=0#Define our best error % based off our linear regression equation
for _ in range(500):#We will run the training 500 times to optimize our equation for the most accurate prediction.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    #Create our model with a size of 10% of our data which is about 6 games

    linear = linear_model.LinearRegression()#Run our linear regression model

    linear.fit(x_train, y_train)#Fit a trendline
    acc = linear.score(x_test, y_test)#Compare our equation against our test values to get an error value

    if acc>best:
        best=acc
    #If our value is higher that means that it was more accurate so this will replace our value with the highest
    #after each run

print("Co: \n",linear.coef_)#Output our information about our linear regression equation
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)#Predict our values in our test set based off of our optimized model

print('{:<8}'.format('Year')+'{:^7}'.format('Wins')+'{:>8}'.format('Prediction'))#Output results
for x in range(len(predictions)):
    print('{:<10}'.format((str(x_test[x][0]))+
    '{:>8}'.format(str(y_test[x]))+'{:>10}'.format(str(int(round(predictions[x]))))))


p = "Div. Finish"#Show a graph displaying Division Finish vs Wins
ax = plt.subplot()
ax.scatter(data[p], data["W"])
ax.invert_xaxis()
ax.set_xlabel(p)
ax.set_ylabel("Wins")
ax.set_title("Wins VS "+p)
plt.show()