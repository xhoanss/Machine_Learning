# id:16-32-16

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

matplotlib.use('TkAgg')
df = pd.read_csv("week3.csv",header=None)
x = df.iloc[:,0] #first column
y = df.iloc[:,1] #second column
z = df.iloc[:,2] #third column
X = np.column_stack((x,y))
xtrain,xtest,ztrain,ztest = train_test_split(X,z,test_size=0.2)
xtrain_poly = PolynomialFeatures(degree=5).fit_transform(xtrain)
xtest_poly = PolynomialFeatures(degree=5).fit_transform(xtest)
Xpoly = PolynomialFeatures(degree=5).fit_transform(X)

def A1():
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel('x1',fontdict={'size':15,'color':'black'})
    ax.set_ylabel('x2', fontdict={'size': 15, 'color': 'black'})
    ax.set_zlabel('y', fontdict={'size': 15, 'color': 'black'})
    plt.show()
    #show as a curve

def B1(dummy):
    #a = 1/2c
    Lmodel = linear_model.Lasso(alpha=0.0005)
    Lmodel.fit(xtrain_poly,ztrain)
    Lmodel_predict = Lmodel.predict(xtest_poly)
    print("accuracy:",Lmodel.score(xtest_poly,ztest))
    print("slope is:",Lmodel.coef_)
    print("intercept is:",Lmodel.intercept_)
    print("square error between test and predict =",mean_squared_error(ztest,Lmodel_predict))
    print("square error between test and dummy =", mean_squared_error(ztest, dummy))

    #for i(c)
    # get the range of prediction
    range = MinMax(Lmodel_predict)
    Xpre = []
    grid = np.linspace(-2,2)
    for i in grid:
        for j in grid:
            Xpre.append([i,j])
    Xpre = np.array(Xpre)
    Xpre_poly = PolynomialFeatures(degree=5).fit_transform(Xpre)
    predictions = Lmodel.predict(Xpre_poly)
    #draw
    fig = plt.figure()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], ztrain,color='red',label ='data')
    ax.plot_trisurf(Xpre[:,0],Xpre[:,1],predictions)
    plt.title('Lasso, C = 1000', fontsize='17')
    ax.set_xlabel('x1', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('x2', fontdict={'size': 15, 'color': 'black'})
    ax.set_zlabel('y', fontdict={'size': 15, 'color': 'black'})
    plt.legend()
    plt.show()

def E1(dummy):
    # a = 1/2c
    Rmodel = linear_model.Ridge(alpha=0.0005)
    Rmodel.fit(xtrain_poly,ztrain)
    Rmodel_predict = Rmodel.predict(xtest_poly)
    print("accuracy:", Rmodel.score(xtest_poly, ztest))
    print("slope is:", Rmodel.coef_)
    print("intercept is:", Rmodel.intercept_)
    # print(Rmodel_predict)
    print("square error between test and predict =", mean_squared_error(ztest, Rmodel_predict))
    print("square error between test and dummy =", mean_squared_error(ztest, dummy))
    # for i(c)
    # get the range of prediction
    range = MinMax(Rmodel_predict)
    Xpre = []
    grid = np.linspace(-2, 2)
    for i in grid:
        for j in grid:
            Xpre.append([i, j])
    Xpre = np.array(Xpre)
    Xpre_poly = PolynomialFeatures(degree=5).fit_transform(Xpre)
    predictions = Rmodel.predict(Xpre_poly)
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], ztrain,color='red',label='data')
    ax.plot_trisurf(Xpre[:, 0], Xpre[:, 1], predictions)
    plt.title('Ridge, C = 1000', fontsize='17')
    ax.set_xlabel('x1', fontdict={'size': 15, 'color': 'black'})
    ax.set_ylabel('x2', fontdict={'size': 15, 'color': 'black'})
    ax.set_zlabel('y', fontdict={'size': 15, 'color': 'black'})
    plt.legend()
    plt.show()


def A2():
    mean = []
    standard = []
    c_range = [1,5,10,20,50,100,500]
    for c in c_range:
        Lmodel = linear_model.Lasso(alpha=1/(2*c))
        kf = KFold(n_splits=5)
        temp = []
        for train,test in kf.split(Xpoly):
            Lmodel.fit(Xpoly[train],z[train])
            y_predict = Lmodel.predict(Xpoly[test])
            temp.append(mean_squared_error(z[test],y_predict))
        mean.append(np.array(temp).mean())
        standard.append(np.array(temp).std())
    print("mean is:",mean)
    print("standard is:",standard)
    fig = plt.figure()
    plt.errorbar(c_range,mean,yerr=standard,ecolor='r')
    plt.title('5 fold cross-validation', fontsize='10')
    plt.xlabel('C =[1,5,10,20,50,100,500]')
    plt.ylabel('Mean Square Error')
    plt.show()

def C2():
    mean = []
    standard = []
    c_range = [1, 5, 10, 20, 50,100,500]
    for c in c_range:
        Rmodel = linear_model.Ridge(alpha=1 / (2 * c))
        kf = KFold(n_splits=5)
        temp = []
        for train, test in kf.split(Xpoly):
            Rmodel.fit(Xpoly[train], z[train])
            y_predict = Rmodel.predict(Xpoly[test])
            temp.append(mean_squared_error(z[test], y_predict))
        mean.append(np.array(temp).mean())
        standard.append(np.array(temp).std())
    print("mean is:", mean)
    print("standard is:", standard)
    fig = plt.figure()
    plt.errorbar(c_range, mean, yerr=standard, ecolor='r')
    plt.title('5 fold cross-validation', fontsize='10')
    plt.xlabel('C =[1,5,10,20,50,100,500]')
    plt.ylabel('Mean Square Error')
    plt.show()


def Baseline():
    from sklearn.dummy import DummyRegressor
    dummy = DummyRegressor(strategy='mean').fit(xtrain_poly,ztrain)
    dummy_predict = dummy.predict(xtest_poly)
    return dummy_predict

def MinMax(arr):
    max = arr[0]
    min = arr[0]
    for i in arr:
        if i > max:
            max = i
        if i < min:
            min = i
    return min,max

if __name__ == '__main__':
    #A1()
    #B1(Baseline())
    E1(Baseline())
    #A2()
    #C2()

