import numpy as np
import tensorflow as tf


class Mlp(tf.keras.Model):
        
    def __init__(self, output_dim=3, hidden_layer_size=1024, num_of_layer=3, keep_prob=1, l2_reg=0):
        super(Mlp,self).__init__()
        #Parameters
        self.hidden_layer_size = hidden_layer_size,
        self.num_of_layer = num_of_layer
        self.l2_reg = l2_reg
        self.keep_prob = keep_prob
        self.output_dim = output_dim
        #Create layers
        self.layers_ = []
        self.layers_dropout = []
        for k in range (self.num_of_layer):
            self.layers_.append(tf.keras.layers.Dense(self.hidden_layer_size[0], activation=tf.nn.relu, name=f"mlp_theta_{k}"))
            self.layers_dropout.append(tf.keras.layers.Dropout(1-keep_prob, name="mlp_theta") )        
        self.layers_.append(tf.keras.layers.Dense( self.output_dim, use_bias=True, activation=None, name=f"mlp_theta_{self.num_of_layer}"))

    def call(self,inputs):
        output = inputs
        if self.num_of_layer > 0 :
            for layer, drop in zip(self.layers_[:-1],self.layers_dropout):
                output = drop(layer(output))
        output = self.layers_[-1](output)
        return output
