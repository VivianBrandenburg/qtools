# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from qtools.encoding import onehot_encoding




#https://github.com/keras-team/keras/pull/10856
#puts four identical models besides each other and concatenates the output. 



class quartetmodel(tf.keras.Model):
    
    
    def __init__(self, *args, **kwargs):
        super(quartetmodel, self).__init__(*args, **kwargs)
  
        if not hasattr(self, 'encoding_function'):
            self.add_model_infos()
    
    
    
    @classmethod
    def from_h5(cls, h5_filepath, loss_function):
        from tensorflow.keras.models import load_model
        qmodel = load_model(h5_filepath, custom_objects={'sf_Xsq':loss_function})
        return cls(inputs=qmodel.input, outputs=qmodel.output)
    
   
    
            
    @classmethod
    def from_basemodel(cls,basemodel):
        """ Takes a Sequential tensorflow model and returns a quartet model
        """
        R=tf.keras.layers.Reshape(basemodel.output_shape[1:]+ (1,),
                                  input_shape=basemodel.output_shape[1:])
        a=tf.keras.Input(basemodel.input.shape[1:])
        b=tf.keras.Input(basemodel.input.shape[1:])
        c=tf.keras.Input(basemodel.input.shape[1:])
        d=tf.keras.Input(basemodel.input.shape[1:])
        o=tf.keras.layers.Concatenate()([R(basemodel(a)), R(basemodel(b)),
                                          R(basemodel(c)),R(basemodel(d))])
        return cls(inputs=[a,b,c,d], outputs=o)
        
    
    @property
    def basemodel(self):
        """ Returns the inner Sequential Model of the quartet / siamese model 
        """
        # model layers should always be [Inputs*n, sequential, reshape, concatenation], thus layer[-3] should always be sequential, independent from number of input layers
        return self.layers[-3] 
    
    def add_model_infos(self):    
        self.input_dims = self.basemodel.input_shape
        self.output_dims = self.basemodel.output_shape
        self.seq_len = self.basemodel.input_shape[-2]
        self.features = self.basemodel.output_shape[-1]
        
        #add the encoding function
        assert self.seq_len != 4, 'Sequence length is 4nt. Cannot determine which encoding function to use.'
        if self.input_dims[-1]==4: self.encoding_function = onehot_encoding
        else: self.encoding_function = 'unknown' 
        
        
        
    def predict(self, data): 
        if isinstance(data, (list, np.ndarray)):
            return self.predict_from_onehot(data)
        else:
            import pandas as pd 
            if isinstance(data, pd.DataFrame):
                return self.predict_from_dataframe(data)
            else:
                raise NotImplementedError('Data can be given as sequence OR encoded in a DataFrame (pandas or qdata) OR encoded in an numpy array. The data I recieved fit none of the above formats.')
                
    def predict_from_onehot(self, x):
        return self.basemodel.predict(np.array(x))
    
    def predict_from_dataframe(self, data):
        if not hasattr(data, 'encoded'):
            data.encode(self.encoding_function)
        data_onehot = np.array(list(data.encoded))
        return self.basemodel.predict(data_onehot)
    
    

    def get_distance_matrix(self, prediction):
        """
        Predict a pairwise distance matrix between two predicted featrue vectors
        """
        matrix = distance_matrix(prediction, prediction, p=1)
        return matrix

    
    def predict_and_plot(self, data, outfile=False, data_labels=False):
            
        """
        Takes data, predicts the features and plots a heatmap of the prediction
    
        Parameters
        ----------
        data : qdata/pandas DataFrame or path/to/csv or numpy array of encoded data
            data to predict. must have 'seq' and 'spec' columns. 
        outfile : path, optional
            Provide filename here if you want the heatmap saved. The default is None / do not save.
        data_labels: list of strings, optional
            provide labels if these are not already included in data
    
        Returns
        -------
        None. Plots in console, can produce plot as file (optional)
    
        """
        import seaborn as sns 
        import matplotlib.pyplot as plt
        from qtools import qdata

      
        # if the data are not read in yet, read them 
        if type(data) == str:
            data = qdata(data, encoding_function=self.encoding_function)
        # if data are in qdata format, get encoded data
        if isinstance(data,qdata):
            data_onehot = np.array(list(data.encoded))
            data_spec = data.spec
        else:
            data_onehot = data
            data_spec = range(len(data))
            if data_labels:
                data_spec = data_labels
            
        # predict model output and strutcure it for seaborn 
        prediction = self.predict(data_onehot)
        prediction_structured = pd.DataFrame(prediction, index=data_spec, columns=range(self.features))
        
        # visualize predictions
        sns.set(rc={'figure.figsize':(11,5.5)})
        sns.heatmap(prediction_structured, cmap='magma_r',  mask=(prediction_structured==0))
        if outfile:
            plt.savefig(outfile)
        plt.show()
    



class siamesemodel(quartetmodel):
      
    @classmethod
    def from_basemodel(cls,basemodel):
        """ Takes a Sequential model and returns it as siamese model. 
        """
        R=tf.keras.layers.Reshape(basemodel.output_shape[1:]+ (1,),
                                  input_shape=basemodel.output_shape[1:])
        a=tf.keras.Input(basemodel.input.shape[1:])
        b=tf.keras.Input(basemodel.input.shape[1:])
        o=tf.keras.layers.Concatenate()([R(basemodel(a)), R(basemodel(b))])
        return cls(inputs=[a,b], outputs=o)
        

    

