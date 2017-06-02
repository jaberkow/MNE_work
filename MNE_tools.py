import numpy as np
import math
from numpy.random import randn
import keras
from keras.models import Model
from keras.layers import Add,Multiply,Dense,Input,Activation
from scipy import ndimage

def block_mean(ar, fact):
    """
    A simple downsampling utility.  I make no claims about the efficiency of this code, but I set it up to be interpretable and easily inspectable
    
    Inputs:
    ar - Two-Dimensional input array of shape (sx,sy)
    fact - downsampling factor.  Must evenly divide both sx and sy
    
    Outputs:
    res2 - Downsampled array of shape (sx/fact,sy/fact)
    """
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    res2 = np.floor(res)
    return res2


def makeModel(N,order=1):
    """
    To run the model, use the following:

    model = makeModel(n_dimensions, order)
    model.fit(X, Y, epochs=n_epochs, callbacks=[keras.callbacks.EarlyStopping()], validation_split=0.25)

    For the linear model, model.get_weights() returns the linear weights and the bias. 
    For the quadratic model, model.get_weights() returns quadratic weights, dummy weights, linear weights, and the bias.
    """
    
    inLayer = Input((N,))
    linear = Dense(1,name='Linear')(inLayer)
    if order == 2:
        quad1 = Dense(N,kernel_initializer=initializers.random_normal(stddev=0.001),use_bias=False,name='Quad')(inLayer)
        quad2 = Multiply()([quad1,inLayer])
        quad3 = Dense(1,kernel_initializer='ones',use_bias=False,trainable=False)(quad2)
        x = Add()([quad3,linear])
        outLayer = Activation('sigmoid')(x)
    else:
        outLayer = Activation('sigmoid')(linear)
    model = Model(inLayer,outLayer)
    model.compile('rmsprop','binary_crossentropy')
    return model

def train_MNE(stimuli,spike_probs,order=2,val_split=0.25):
    """
    Creates and fits a MNE model
    
    Inputs:
    stimuli - numpy array of the stimuli.  Of shape (number of samples,dimension of stimulus)
    spike_probs - numpy array of the spiking probabilities.  Of shape (number of samples)
    order - The order of the MNE fit.  1 = first order linear filters only (standard logistic regression).  2 = also fit second  order quadratic kernels.
    val_split - what percentage of data to set aside for early stopping on the validation set.
    
    Outputs:
    results - A python dictionary with resulting fitted parameters.
    """
    
    
    dim_x = np.shape(stimuli)[1]
    model = makeModel(dim_x, order=order)
          
    model.fit(stimuli,spike_probs,verbose=0,epochs=40,callbacks=[keras.callbacks.EarlyStopping(patience=2)],validation_split=val_split)
    if order==1:    
        w,bias = model.get_weights()
        results = {"linear":w,"bias":bias}
    else:
        J,dummy,w,bias = model.get_weights()
        J_sym = 0.5*(J + np.transpose(J))
        results = {"quadratic":J_sym,"linear":w,"bias":bias}
    return results

