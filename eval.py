import pickle 
import numpy as np 
from sklearn.svm import SVC 
from sklearn.grid_search import GridSearchCV   
from keras.datasets import mnist
from sklearn.metrics import mean_squared_error, accuracy_score

def main():
    """ Test SVM from scikit learn on mnist data set.""" 

    (X_train, Y_train), (X_test, Y_test) =  mnist.load_data() 
  
    # preprocess data
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


    model = SVC(kernel='rbf', gamma=0.02, C=10) 
    model.fit(X_train, Y_train)
    
    train_yy = model.predict(X_train)
    test_yy = model.predict(X_test) 

    train_err = 100*mean_squared_error(train_yy, Y_train) 
    test_err = 100*mean_squared_error(test_yy, Y_test) 
    
    print("Train. err:", train_err) 
    print("Test err:", test_err) 

    train_acc = accuracy_score(Y_train, train_yy)  
    test_acc = accuracy_score(Y_test, test_yy) 

    pickle.dump(model, open("svm_rbf", "wb"))

def test():
    (X_train, Y_train), (X_test, Y_test) =  mnist.load_data() 
  
    # preprocess data
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    model = pickle.load(open("svm_rbf.pickle","rb"))
    
    train_yy = model.predict(X_train)
    test_yy = model.predict(X_test) 

    train_err = 100*mean_squared_error(train_yy, Y_train) 
    test_err = 100*mean_squared_error(test_yy, Y_test) 
    
    print("Train. err:", train_err) 
    print("Test err:", test_err) 

    train_acc = accuracy_score(Y_train, train_yy)  
    test_acc = accuracy_score(Y_test, test_yy) 
    
    print("Train acc:", train_acc)
    print("Test acc:", test_acc)


if __name__=="__main__": 
    test() 
