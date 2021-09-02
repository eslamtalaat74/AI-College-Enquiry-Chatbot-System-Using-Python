

import tflearn                                #TFlearn is a modular and transparent deep learning library built on top of Tensorflow.
                                              #used for provide a higher level API to tensorflow for facilitating and showing up new experiments
from tensorflow.python.framework import ops
#Developing a Model
def init_model(X_train,y_train,X_test,y_test):       
    
    ops.reset_default_graph()                            #for removing any defalts that were pefor       
                                                          #use a fairly standard feed-forward neural network with two hidden layers. 
                                                         # The goal of our network will be to look at a bag of words and give a class that they 
                                                         # ong too
    net = tflearn.input_data(shape=[None, len(X_train[0])])    #input layer ,the size of the layer is the same size as x training data list
    net = tflearn.fully_connected(net, 550)                       # two hidden layers  each one is made up of 550 neruns
    net = tflearn.fully_connected(net, 550)
    net = tflearn.fully_connected(net, len(y_train[0]), activation = "softmax")   #output layer according to the length of the y train list
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    
    import os
    if os.path.exists("model.tflearn.meta"):     #if the model already exists ;load this model 
        model.load("model.tflearn")
    else:                                        #else start training from the first 
        model.fit(X_train, y_train,validation_set=(X_test,y_test), n_epoch=10, batch_size=8, show_metric=True)  #training our datset
                                                                                                                 #epch =500 : the number of
                                                                                                                 #  times we will train 
                                                                                                                 # (see data)
                                                                                                                  #batch size : the number 
                                                                                                                  # of inputes that takes at 
                                                                                                                  # one time
                                                                                                                  #show metric to show accurecy 
                                                                                                                  # , losses ...etc
        model.save("model.tflearn")                                                                              #save our model
        
    return model
