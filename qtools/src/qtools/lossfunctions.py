#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.losses import Loss



# =============================================================================
# distance calculations 
# =============================================================================
    
def computedistancematrix(y_pred, norm):
    Dij=norm(y_pred[:,:,0]-y_pred[:,:,1])
    Dik=norm(y_pred[:,:,0]-y_pred[:,:,2])
    Dil=norm(y_pred[:,:,0]-y_pred[:,:,3])
    Djk=norm(y_pred[:,:,1]-y_pred[:,:,2])
    Djl=norm(y_pred[:,:,1]-y_pred[:,:,3])
    Dkl=norm(y_pred[:,:,2]-y_pred[:,:,3])
    return Dij, Dik, Dil, Djk, Djl, Dkl

def get_distance_matrix(y_pred):
    ypred_dist = computedistancematrix(y_pred, lambda x: tf.norm(x, ord='euclidean', axis=-1))
    return ypred_dist



# =============================================================================
# quartet loss and siamese regulation 
# =============================================================================


class Xsq_SiamReg(Loss):
    
    
    def __init__(self, sigma):
        """
        Initialize the quartet loss function and set sigma. 
        Sigma is the relation between quartet loss and siamese regulation. 
        
        Loss = Quartet loss + (Siamese Regulation * Sigma)

        """
        super().__init__()
        self.sigma = tf.constant(float(sigma))
    
    
    def call(self, y_true,y_pred):  
    
            
        #####  e-loss  ####
        e_loss = eloss(y_true,y_pred)
    
        ##### siamese regulation  ####
        if self.sigma !=  0: 
            siam_loss = siamloss(y_true,y_pred) 
            loss = tf.add(e_loss, tf.multiply(siam_loss, self.sigma))
        
        else:
            loss = e_loss
            
        return loss
    



def eloss(y_true,y_pred):
    # calculate distances between output feature vectors
    ypred_dist = get_distance_matrix(y_pred)
    
    # split distances between feature vectors
    Dij, Dik, Dil, Djk, Djl, Dkl= tf.split(ypred_dist, num_or_size_splits=6, axis=0)

    #####  e-loss  ####
    e = tf.subtract( tf.add(Dil,Djk) , tf.add(Dik,Djl))
    e_loss = tf.square(e)
    return e_loss

   

def siamloss(y_true,y_pred):  
    # calculate distances between output feature vectors
    ypred_dist = get_distance_matrix(y_pred)
    
    # difference between predicted distance and pairwise sequence distance
    diff = tf.abs(tf.subtract(y_true, ypred_dist))
    siam_loss = tf.reduce_sum(diff)
    return siam_loss





# =============================================================================
# loss for siamese network 
# =============================================================================





def siamloss_siamnet(y_true,y_pred):  
    # calculate distances between output feature vectors
    norm = lambda x: tf.norm(x, ord='euclidean', axis=-1)
    Dij=norm(y_pred[:,:,0]-y_pred[:,:,1])  
    #calculate difference ytrue and ypredict
    siam_loss = tf.abs(tf.subtract(y_true, Dij))
    return siam_loss


