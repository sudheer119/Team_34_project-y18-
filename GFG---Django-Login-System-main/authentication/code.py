import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as mlt
import seaborn as sn
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import networkx as nx

# data set cretaed

dataset = pd.read_csv("C:/Users/nsudh/Desktop/BUS.csv")
X = dataset[dataset.columns[:-1]]

y = dataset.iloc[:, -1].values


# Train and split the code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

reg1 = linear_model.LinearRegression()
reg2 = linear_model.BayesianRidge()
reg3 = svm.SVR()
reg4 = tree.DecisionTreeRegressor()
# reg5 = GradientBoostingRegressor(random_state=1)
# reg6 = RandomForestRegressor(random_state=1)
#reg7 = MLPRegressor(random_state=1)

# fit the  data
reg1.fit(X_train, y_train)
reg2.fit(X_train, y_train)
reg3.fit(X_train, y_train)
reg4.fit(X_train, y_train)
# reg5.fit(X_train, y_train)
# reg6.fit(X_train, y_train)
#reg7.fit(X_train, y_train)

# prediction
y_pred1 = reg1.predict(X_test)
y_pred2 = reg2.predict(X_test)
y_pred3 = reg3.predict(X_test)
y_pred4 = reg4.predict(X_test)
# y_pred5 = reg5.predict(X_test)
# y_pred6 = reg6.predict(X_test)
#y_pred7 = reg7.predict(X_test)


def cal(num1, num2, num3, num4, num5):
    event = int(num1)
    startloaction = int(num2)
    endlocation = int(num3)
    season = int(num4)
    time = int(num5)
    c = []
    if startloaction != endlocation:
        a = reg1.predict([[event, startloaction, endlocation, season, time]])
        if a[0] <= 50:
            return "DO NOT RUN THE BUS in this route dur to less tickets can be sold : " + str(round(a[0], 2))
        else:
            c = "No of tickets can be booked in that rute if you run is " + \
                str(round(a[0], 2))+"  evnet : "+num1+" startloaction :" + \
                num2+" endlocation: "+num3+" season: "+num4+" time: "+num5
            return c
    else:
        c="strat location wnd location cannot be same :"
        return c
