# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:20:41 2019

@author: Mike
"""
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers, optimizers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping


# In *older* versions of Tensorflow/Keras you may need to adjust the image
# dimension ordering. Read about channel ordering here:
#    https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/

# This is the main function. You need to write the getModel and fitModel functions to pass to this.
# Call your functions 'myGetModel' and 'myFitModel'.
# The getModel function should accept an object of the CIFAR class, and return a compiled Keras CNN model.
# In this function you will specify the network structure (including regularization) and the optimizer to
# be used (and its parameters like learning rate), and run compile the model (in the Keras sense of running
# model.compile).
# The fitModel function should accect two arguments. The first is the CNN model you return from your getModel
# function, and the second is the CIFAR classed data object. It will return a trained Keras CNN model, which
# will then be applied to the test data. In this function you will train the model, using the Keras model.fit
# function. You will need to specify all parameters of the training algorithm (batch size, etc), and the
# callbacks you will use (EarlyStopping and ModelCheckpoint). You will need to make sure you save and load
# into the model the weight values of its best performing epoch.
def runImageClassification(getModel=None,fitModel=None,seed=7):
    # Fetch data. You may need to be connected to the internet the first time this is done.
    # After the first time, it should be available in your system. On the off chance this
    # is not the case on your system and you find yourself repeatedly downloading the data,
    # you should change this code so you can load the data once and pass it to this function.
    print("Preparing data...")
    data=CIFAR(seed)

    # Create model
    print("Creating model...")
    model=getModel(data)

    # Fit model
    print("Fitting model...")
    model=fitModel(model,data)

    # Evaluate on test data
    print("Evaluating model...")
    score = model.evaluate(data.x_test, data.y_test, verbose=0)
    print('Test accuracy:', score[1])

# This is the class that wraps the CIFAR data. You will probably need to be connected to the
# internet the first time you create an object of this class, as the data will be downloaded.
# After that, the data should be stored by Keras and no downloading will be required.
# Important fields that you will need to use are: x_train, y_train, x_valid, y_valid, input_dim and
# num_classes. The first four of these are the training and validation data (split into features and
# target). Note that these have been made ready for use with a Keras network - check out the code
# if you are interested. The last two are the number of input features and the number of target
# classes. These will be needed when defining your CNN.
# The only public method is the showImages function, which you can use to see some labelled images
# from the (validation) data.
# Remember that the x_test and y_test fields will be blank when your functions are run in evaluation -
# so you cannot peek at these cases!
class CIFAR:
    def __init__(self,seed=0):
        # Get and split data
        data = self.__getData(seed)
        self.x_train_raw=data[0][0]
        self.y_train_raw=data[0][1]
        self.x_valid_raw=data[1][0]
        self.y_valid_raw=data[1][1]
        self.x_test_raw=data[2][0]
        self.y_test_raw=data[2][1]
        # Record input/output dimensions
        self.num_classes=10
        self.input_dim=self.x_train_raw.shape[1:]
         # Convert data
        self.y_train = np_utils.to_categorical(self.y_train_raw, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid_raw, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_raw, self.num_classes)
        self.x_train = self.x_train_raw.astype('float32')
        self.x_valid = self.x_valid_raw.astype('float32')
        self.x_test = self.x_test_raw.astype('float32')
        self.x_train  /= 255
        self.x_valid  /= 255
        self.x_test /= 255
        # Class names
        self.class_names=['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    def __getData (self,seed=0):
        #to download data from cifar, update the certifi to get access to the data
        #pip3 install certifi
        #/Applications/Python 3.6/Install Certificates.command
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return self.__shuffleData(x_train,y_train,x_test,y_test,seed)

    def __shuffleData (self,x_train,y_train,x_test,y_test,seed=0):
        tr_perc=.75
        va_perc=.15
        x=np.concatenate((x_train,x_test))
        y=np.concatenate((y_train,y_test))
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        indices = np.random.permutation(len(x))
        tr=round(len(x)*tr_perc)
        va=round(len(x)*va_perc)
        self.tr_indices=indices[0:tr]
        self.va_indices=indices[tr:(tr+va)]
        self.te_indices=indices[(tr+va):len(x)]
        x_tr=x[self.tr_indices,]
        x_va=x[self.va_indices,]
        x_te=x[self.te_indices,]
        y_tr=y[self.tr_indices,]
        y_va=y[self.va_indices,]
        y_te=y[self.te_indices,]
        return ((x_tr,y_tr),(x_va,y_va),(x_te,y_te))

    # Print 25 random figures from the validation data
    def showImages(self):
        images=self.x_valid_raw
        labels=self.y_valid_raw
        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        plt.figure(figsize=(10,10))
        indices=np.random.randint(0,images.shape[0],25)
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[indices[i]], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays,
            # which is why we need the extra index
            plt.xlabel(class_names[labels[indices[i]][0]])
        plt.show()


#input: an object of the CIFAR class
#output: a compiled Keras CNN model
#content: to specify the netwrok structure(including regularization)
#         and the optimizer to be used(including learning rate)
#         and run compile the model (in the Keras sense of running model compile).
def myGetModel(cifar):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=cifar.x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(cifar.num_classes))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Use multinomial logistic regression
    model.add(Activation('softmax'))


    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Compile - SGD, Loss function, Performance Metric
    # Logarithmic loss, categorical cross entropy
    # Metrics is the accuracy metric
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


#input: CNN model from myGetModel; CIFAR classed data object
#output: a trained Keras CNN model, which will then be applied to the test data
#content: train the model, using the Keras mdoel.fit function
#         to specify all parameters of the training algorithms(batch size, etc)
#         the callbacks: earlyStopping and ModelCheckpoint
#         save and load into the model the weight values of its best performing epoch.
def myFitModel(model, data):


    # Use early stopping to stop training when the loss function starts to increase
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None,
                                  restore_best_weights=False)
    checkpointer = ModelCheckpoint(
        filepath='weights_best.hdf5',
        verbose=1, save_best_only=True)
    model.fit(data.x_train, data.y_train, batch_size=32, epochs=4, validation_split=0.1,
              callbacks=[earlyStopping, checkpointer])
    return model



def main():
    runImageClassification(myGetModel,myFitModel)


if __name__ == '__main__':
    main()
