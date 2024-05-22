import numpy as np
import tensorflow as tf


class Cdgn_Mlp(tf.keras.Model):
        
    def __init__(self,
                 output_dim = 3,
                 hidden_layer_size  = 1024,
                 num_of_layer = 3,
                 keep_prob = 1,
                 use_common_hidden_layer = True,
                 l2_reg = 0,
                 ):
        super(Cdgn_Mlp,self).__init__()
        #Parameters
        self.hiden_layer_size = hidden_layer_size,
        self.nb_of_layer = num_of_layer
        self.l2_reg = l2_reg
        self.keep_prob = keep_prob
        self.use_common_hidden_layer = use_common_hidden_layer
        self.output_dim = output_dim
        #Create layers
        if self.use_common_hidden_layer :
            self.layers_ = [ tf.keras.layers.Dense( self.hiden_layer_size[0], activation = tf.nn.relu, name = "cdgn_phi" )  for i in range( self.nb_of_layer ) ]
            self.layers_dropout = [tf.keras.layers.Dropout(1-keep_prob, name = "cdgn_phi") for i in range( self.nb_of_layer  ) ]
        else :
            self.layers_mean = [tf.keras.layers.Dense( self.hiden_layer_size[0], activation = tf.nn.relu, name = "cdgn_phi" )  for i in range( self.nb_of_layer ) ]
            self.layers_mean_dropout = [tf.keras.layers.Dropout(1-keep_prob, name = "cdgn_phi") for i in range( self.nb_of_layer  ) ]
            self.layers_logvar = [tf.keras.layers.Dense( self.hiden_layer_size[0], activation = tf.nn.relu, name = "cdgn_phi" )  for i in range( self.nb_of_layer ) ]
            self.layers_logvar_dropout = [tf.keras.layers.Dropout(1-keep_prob, name = "cdgn_phi") for i in range( self.nb_of_layer  ) ] 
        self.mean = tf.keras.layers.Dense( self.output_dim, activation = None, kernel_initializer = "zeros", name = "cdgn_phi")
        self.logvar =  tf.keras.layers.Dense( self.output_dim, activation = None, kernel_initializer = "zeros", name = "cdgn_phi" )
        
    def call(self,inputs):
        if self.use_common_hidden_layer :
            output = tf.concat(inputs, axis = 1)
            for layer, drop in zip(self.layers_,self.layers_dropout):
                output = drop(layer(output))
            return self.mean(output), self.logvar(output)
        else :
            output_mean = tf.concat(inputs, axis = 1)
            output_logvar = tf.concat(inputs, axis = 1)
            for layer_mean, layer_logvar, drop_mean, drop_logvar in zip(self.layers_mean, 
                                                                        self.layers_logvar, 
                                                                        self.layers_mean_dropout, 
                                                                        self.layers_logvar_dropout ):
                output_mean = drop_mean(layer_mean(output_mean))
                output_logvar = drop_logvar(layer_logvar(output_logvar))
            return self.mean(output_mean), self.logvar(output_logvar)