from keras.datasets import mnist 
import scipy.fftpack as ft

nb_classes = 10 

def preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return (X_train, Y_train), (X_test, Y_test) 


def add_tranform_to_dataset(dataset):
    """Adds cosine transform to each sample """
    
    transformed = [] 
    for x in dataset:
        xt = ft.dct(x) 
        transformed.append(xt[:,:14]) 

    return np.hstack((dataset, transformed)) 

def preprocess_mnist_cosin():
    (X_train, Y_train), (X_test, Y_test) = preprocess_mnist() 

    X_train2 = add_transform_to_dataset(X_train) 
    X_test2 = add_transform_to_dataset(X_test) 

    return (X_train2, Y_train), (X_test2, Y_test)
        
