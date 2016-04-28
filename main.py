import numpy as np 
from sklearn.svm import SVC 
from sklearn.grid_search import GridSearchCV   
from keras.datasets import mnist


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


    model = SVC(kernel='rbf') 
    params = { "C" : np.logspace(-7, 0, 4), 
               "gamma" : np.logspace(-3, 0, 4) } 

    grid = GridSearchCV(model, param_grid = params, 
                        cv=5, n_jobs = 5, pre_dispatch = "n_jobs")
    grid.fit(X_train, Y_train) 

    print(grid.best_params_)


if __name__=="__main__": 
    main() 
