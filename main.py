# id:9-9-9-0
# id:9--9-9-0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.dummy import DummyClassifier


matplotlib.use('TkAgg')
df1 = pd.read_csv("week4.csv")
df1.head()
df1.columns = ['x1', 'x2', 'y']
data1_x1 = df1.iloc[:,0] #first column
data1_x2 = df1.iloc[:,1] #second column
data1_y = df1.iloc[:,2] #third column
data1_X = np.column_stack((data1_x1,data1_x2))

df2 = pd.read_csv("week4_2.csv")
df2.head()
df2.columns = ['x1', 'x2', 'y']
data2_x1 = df2.iloc[:,0] #first column
data2_x2 = df2.iloc[:,1] #second column
data2_y = df2.iloc[:,2] #third column
data2_X = np.column_stack((data2_x2,data2_x2))

###################
# Visualisation Data
###################
def visualisation_Data1(df):
    plt.rc('font', size=10)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(df.loc[df['y'] == 1, 'x1'], df.loc[df['y'] == 1, 'x2'], color='red', marker='+', label='training data y=1')
    plt.scatter(df.loc[df['y'] == -1, 'x1'], df.loc[df['y'] == -1, 'x2'], color='blue', marker='+', label='training data y=-1')
    plt.legend()
    plt.xlabel("first feature x1")
    plt.ylabel("second feature x2")
    plt.show()
####################
# For A to choose maximum order of polynomial
####################
def choose_p(X,y,c,p_range):
    mean=[]
    str=[]
    model = LogisticRegression(penalty='l2',C=c)

    for p in p_range:
        Xpoly = PolynomialFeatures(p).fit_transform(X)
        scores = cross_val_score(model,Xpoly,y,cv=5,scoring='f1')
        mean.append(np.array(scores).mean())
        str.append(np.array(scores).std())
    plt.errorbar(p_range,mean,yerr=str)
    plt.xlabel('p range')
    plt.ylabel('F1 score')
    plt.title('Choose p when C={}'.format(c))
    plt.show()

####################
# For A to choose C
####################
def choose_c(X,y,c_range,p):
    mean=[]
    str=[]
    Xpoly = PolynomialFeatures(p).fit_transform(X)

    for c in c_range:
        model = LogisticRegression(penalty='l2', C=c)
        scores = cross_val_score(model,Xpoly,y,cv=5,scoring='f1')
        mean.append(np.array(scores).mean())
        str.append(np.array(scores).std())

    plt.errorbar(c_range,mean,yerr=str)
    plt.xlabel('C range')
    plt.ylabel('F1 score')
    plt.title('Choose C when p={}'.format(p))
    plt.show()
########################
# For B to choose K
########################
def choose_k(X,y,k_range):
    kf = KFold(n_splits=5)
    mean_error=[]
    str_error=[]

    for k in k_range:
        mse_collect=[]
        model = KNeighborsClassifier(n_neighbors=k,weights='uniform')

        for train,test in kf.split(X):
            model.fit(X[train],y[train])
            y_predict = model.predict(X[test])
            mse = mean_squared_error(y[test],y_predict)
            mse_collect.append(mse)

        mean_error.append(np.array(mse_collect).mean())
        str_error.append(np.array(mse_collect).std())

    plt.errorbar(k_range,mean_error,yerr=str_error,color='red')
    plt.xlabel('k range')
    plt.ylabel('Mean Square Error')
    plt.title('KNN')
    plt.show()

########################
# For C
########################
def con_matrix(y,predict):
    print(confusion_matrix(y,predict))

######################
# For D
######################
def roc(Lmodel,Kmodel,Dmodel,X,y):
    #Logistic model
    Xpoly = PolynomialFeatures(2).fit_transform(X)
    score1 = Lmodel.decision_function(Xpoly)
    fpr,tpr,_ = roc_curve(y,score1)
    plt.plot(fpr,tpr,color='r',label = 'Logistic Regression')

    #KNN model
    score2 = Kmodel.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, score2[:,1])
    plt.plot(fpr, tpr, color='blue', label='KNN')

    #BaseLine
    score3 = Dmodel.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, score3[:, 1])

    plt.plot(fpr, tpr, color='orange', label='Dummy')

    plt.plot([0, 1], [0, 1], 'g--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')

    plt.legend(['Logistic Regression', 'knn', 'Baseline ', 'Random Classifier'])
    plt.show()

#####################
# For final Logistic model to predict and draw
#####################
def final_Lmodel(p,df,c,y,X):
    Xpoly = PolynomialFeatures(p).fit_transform(X)
    Lmodel = LogisticRegression(penalty='l2', C=c)
    Lmodel.fit(Xpoly, y)
    y_prediction = Lmodel.predict(Xpoly)
    df['Lpredict'] = y_prediction

    #draw
    plt.rc('font', size=10)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(df.loc[df['y'] == 1, 'x1'], df.loc[df['y'] == 1, 'x2'], color='red', marker='o',s=28,
                label='training data y=1')
    plt.scatter(df.loc[df['y'] == -1, 'x1'], df.loc[df['y'] == -1, 'x2'], color='blue', marker='o',s=28,
                label='training data y=-1')
    plt.scatter(df.loc[df['Lpredict'] == 1, 'x1'], df.loc[df['Lpredict'] == 1, 'x2'], color='green', marker='+',s=24,
                label='predict data y=1')
    plt.scatter(df.loc[df['Lpredict'] == -1, 'x1'], df.loc[df['Lpredict'] == -1, 'x2'], color='orange', marker='+',s=24,
                label='predict data y=-1')
    plt.legend()
    plt.xlabel("first feature x1")
    plt.ylabel("second feature x2")
    plt.show()
    return y_prediction,Lmodel

#####################
# For final KNN model to predict and draw
#####################
def final_Kmodel(k,X,y,df):
    Kmodel = KNeighborsClassifier(n_neighbors=k,weights='uniform')
    Kmodel.fit(X,y)
    y_predict = Kmodel.predict(X)
    df['Kpredict'] = y_predict

    #draw
    plt.rc('font', size=10)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(df.loc[df['y'] == 1, 'x1'], df.loc[df['y'] == 1, 'x2'], color='red', marker='o', s=28,
                label='training data y=1')
    plt.scatter(df.loc[df['y'] == -1, 'x1'], df.loc[df['y'] == -1, 'x2'], color='blue', marker='o', s=28,
                label='training data y=-1')
    plt.scatter(df.loc[df['Kpredict'] == 1, 'x1'], df.loc[df['Kpredict'] == 1, 'x2'], color='green', marker='+', s=24,
                label='predict data y=1')
    plt.scatter(df.loc[df['Kpredict'] == -1, 'x1'], df.loc[df['Kpredict'] == -1, 'x2'], color='orange', marker='+',
                s=24,
                label='predict data y=-1')
    plt.legend()
    plt.xlabel("first feature x1")
    plt.ylabel("second feature x2")
    plt.show()
    return y_predict,Kmodel

def baseLine(X,y):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X,y)
    predict = dummy.predict(X)
    return predict,dummy


if __name__ == '__main__':
    ##########
    #visualisation_Data1(df1)
    ################

    ###############
    # For A choose polynomial
    p_range=[1,2,3,4,5,6]
    c = 1 #10,100,1000
    #choose_p(data1_X,data1_y,c,p_range)
    #################

    ###############
    # For A choose C
    p=2
    c_range = [0.1,1,5,10,50,100,500]
    #c_range=[0.1,1,2,3,4,5,6,7,8,9,10]
    #choose_c(data1_X, data1_y, c_range,p)
    #################

    ##################
    #Final logistic Model p=2,c=5 and print matrix
    #Lpredict = final_Lmodel(2,df1,5,data1_y,data1_X)
    #con_matrix(data1_y,Lpredict[0])
    ##################

    #################
    # For B choose K
    #k_range=[5,6,7,8,10,15,20,25,50,100,150]
    k_range=[7,8,9,10,13,14,15,17,20,23,24,25,26,27,28,30]
    #choose_k(data2_X,data2_y,k_range)
    #################

    ################
    # For KNN final model k=25 and print matrix
    #Kpredict = final_Kmodel(25,data1_X,data1_y,df1)
    #con_matrix(data1_y, Kpredict[0])
    ################

    ################
    # For dummy print matrix
    #Dpredict = baseLine(data1_X,data1_y)
    #con_matrix(data1_y,Dpredict[0])
    ################

    #roc(Lpredict[1],Kpredict[1],Dpredict[1],data1_X,data1_y)


