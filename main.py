import numpy as np 
from sklearn.svm import SVC 
from sklearn.grid_search import GridSearchCV   
from sklearn.metrics import mean_squared_error, accuracy_score
import data

def main():
    """ Test SVM from scikit learn on mnist data set.""" 

    (X_train, Y_train), (X_test, Y_test) =  data.preprocess_mnist() 
  

    model = SVC(kernel='poly', degree=2) 
    params = { "C" : np.logspace(0, 3, 4), 
               "gamma" : np.logspace(-7, 2, 4), 
               "coef0" : np.logspace(-4,4,4)} 

    grid = GridSearchCV(model, param_grid = params, 
                        cv=5, n_jobs = 5, pre_dispatch = "n_jobs")
    grid.fit(X_train, Y_train) 
    print(grid.best_params_)

    train_yy = grid.predict(X_train)
    test_yy = grid.predict(X_test) 

    train_err = 100*mean_squared_error(train_yy, Y_train) 
    test_err = 100*mean_squared_error(test_yy, Y_test) 
    
    print("Train. err:", train_err) 
    print("Test err:", test_err) 

    train_acc = accuracy_score(Y_train, train_yy)  
    test_acc = accuracy_score(Y_test, test_yy) 

    print("Train. acc:", train_acc) 
    print("Test acc:", test_acc) 


if __name__=="__main__": 
    main() 
