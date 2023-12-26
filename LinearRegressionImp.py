import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    X_train_add1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add a column of 1 to X matrix

    X_trans = np.transpose(X_train_add1)  # Find the transpose of X
    Xt_X_inv = np.linalg.pinv(np.dot(X_trans, X_train_add1))  # Find the inverse of (XT X)
    Xt_X_inv_Xt = np.dot(Xt_X_inv, X_trans)  # (XT X)^-1 XT
    w = np.dot(Xt_X_inv_Xt, y_train)  # Compute the weight

    return w

def mse(X_train,y_train,w):
    sum = 0;
    N = X_train.shape[0]  # Number of data entries
    X_train_add1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add a column of 1 to X matrix

    for k in range(0, N):  # Loop through all the data entries
        diff = (pred(X_train_add1[k], w) - y_train[k])  # The difference between predicted y and observed y
        sum += diff * diff  # ( y - yi )^2

    return sum / N  # average error

def pred(X_train,w):
    return np.dot(X_train,w) # The value of XW = y (predicted)

def test_SciKit(X_train, X_test, Y_train, Y_test):
    Linear_Reg = linear_model.LinearRegression()
    Linear_Reg.fit(X_train, Y_train)

    LR_predict = Linear_Reg.predict(X_test)

    return mean_squared_error(LR_predict, Y_test)

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)
#Our implementation error is very close to SciKit
#Mean squared error from Part 2a is  2993.112494132217 (Our Implementation)
#Mean squared error from Part 2b is  2993.112494132216 (SciKit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()
