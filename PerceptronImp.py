import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):
    N = X_train.shape[0]  # Number of data
    d = X_train.shape[1]  # dimension of x
    w_best = np.ones(d + 1)  # Create a w_best vector with d+1 rows and initialize it with 1 s
    err_best = 1  # set initial error
    w_temp = w_best  # initialize w_temp to be w0 which is all ones.
    err_temp = 1
    errorIndex = []

    for i in range(0, 5000):  # The iteration of pocket algorithm, set to 5000 as the lab specified

        if (err_temp == 0):  # if error is zero, w_temp is the best weight vector
            return w_temp

        # if error is not zero, perform update on w_temp
        # First Looking for a mis-classified pair of x and y
        X_train_add1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add a column of 1 to X_train

        for k in range(N):
            if (pred(X_train_add1[k], w_temp) != y_train[k]):
                errorIndex.append(k)  # record the index all the misclassified pair

        rm_error_point = np.random.randint(0, len(errorIndex))  # randomly select a misclassified pair

        ## Update w_temp according to the mis-classified pair, wt+1 <--- wt + yk xk
        # yk is y_train[errorIndex[rm_error_point]]
        # xk is X_train_add1[errorIndex[rm_error_point]]
        w_temp = np.add(w_temp,np.multiply(y_train[errorIndex[rm_error_point]], X_train_add1[errorIndex[rm_error_point]]))

        err_temp = errorPer(X_train_add1, y_train, w_temp)  # calculate the new error_temp using w_temp
        errorIndex.clear()  # Clear the array that stores error indexes

        if (abs(err_temp) < err_best):  # if the new error is smaller than error best, it means w_temp is a better weight than w_best
            w_best = w_temp  # put w_temp into the pocket
            err_best = err_temp  # record the error_best for future comparison of error_temp

    return w_best

    # A Version of updating w without randomly picking pairs. (Much Faster but suspected to have bias)
    # for k in range(0,X_train.shape[0]):

    #   if np.sign(np.dot( X_train_add1[k] , w_temp)) != y_train[k]: # if the sign(wXk) is not equal to y sign, means xk and yk is a mis-classified point
    #     w_temp = np.add( w_temp , np.multiply( y_train[k] , X_train_add1[k] ) ) # Update w_temp according to the mis-classified pair, wt+1 <--- wt + yk xk
    #     #print("w_temp is :", w_temp)
    #     #print("xk :", X_train_add1[k] , " + yk :", y_train[k])
    #     #print("y times x :", np.multiply(y_train[k], X_train_add1[k] ))
    #     break

def errorPer(X_train,y_train,w):
    sum = 0
    N = X_train.shape[0]

    for k in range(0, N):
        if (pred(X_train[k], w) != y_train[k]):  # if the sign(wXk) is not equal to y sign, means xk and yk is a mis-classified point
            sum += 1  # misclassified point +1

    return sum / N  # return the average amount of mis-classified points

def confMatrix(X_train,y_train,w):
    X_train_add1 = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add a column of 1 before X_train
    conf_matrix = [[0, 0], [0, 0]]
    # conf_matrix[0][0],   conf_matrix[0][1],   conf_matrix[1][0],   conf_matrix[1][1]
    # True Negative     False Positive     False Negative    True Positive
    # P: -1 L: -1     P: +1 L: -1      P: -1 L: +1    P: +1 L: +1

    for k in range(0, X_train.shape[0]):  # go through all rows of x,y pairs
        if (pred(X_train_add1[k], w)) == -1:  # P = -1
            if (y_train[k] == -1):
                conf_matrix[0][0] += 1  # L: -1, True Negative
            else:
                conf_matrix[1][0] += 1  # L: +1, False Negative

        else:  # P = +1
            if (y_train[k] == +1):
                conf_matrix[1][1] += 1  # L: +1, True Positive
            else:
                conf_matrix[0][1] += 1  # L: -1, False Positive

    return conf_matrix

def pred(X_train,w):
    classify = np.sign(np.dot(w, X_train))  # get the sign of the w dot xk row

    if (classify == 0):  # only strictly positive classify goes to +1
        return -1

    return classify

def test_SciKit(X_train, X_test, Y_train, Y_test):
    pct = Perceptron(max_iter=5000, n_iter_no_change=50)
    pct.fit(X_train, Y_train)
    predicted_pct = pct.predict(X_test)
    return confusion_matrix(Y_test, predicted_pct)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
