import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from layers.mlp import Mlp
from layers.cdgn_mlp import Cdgn_Mlp
from layers.cdgn_krylov import Cdgn_Krylov



class MVAE_PoE_Private(tf.keras.Model):
    def __init__(self, 
                 latent_dim=3,
                 num_of_z_sampled=1, 
                 num_of_layer=3,
                 hidden_layer=1024,
                 keep_prob=0.5, 
                 encoder_use_common_hidden_layer=True,
                 decoder_scalar_std=True,
                 use_gcn=False,
                 l2_reg=0
                ):
        super(MVAE_PoE_Private, self).__init__() 
        #Parameters
        self.latent_dim = latent_dim
        self.num_of_z_sampled = num_of_z_sampled
        self.num_of_layer = num_of_layer
        self.hidden_layer = hidden_layer
        self.keep_prob = keep_prob
        self.encoder_use_common_hidden_layer = encoder_use_common_hidden_layer
        self.decoder_scalar_std = decoder_scalar_std
        self.use_gcn = use_gcn
        self.l2_reg = l2_reg
        
    def build(self, shape):
        self.num_of_views = len(shape)
        self.views_features_dimension = [s[1] for s in shape]
        self.coeffnormalisation = [self.views_features_dimension[s] * tf.math.log(2*np.pi ) for s in range(self.num_of_views)]
        # Decoders mean: conditional gaussian distribution with MLP
        self.cdgn_decoders_mean = [Mlp(output_dim=self.views_features_dimension[s],
                                       num_of_layer=self.num_of_layer,
                                       hidden_layer_size=self.hidden_layer*2, # TODO new modification
                                       keep_prob=self.keep_prob,
                                       l2_reg=self.l2_reg)
                                for s in range(self.num_of_views)]
        # Decoders std: weights matrix
        if self.decoder_scalar_std:
            self.rPhi = [self.add_weight(shape=(1, 1), initializer=tf.initializers.Ones(), 
                                        name=f"decoder_scalar_std_{s}") for s in range(self.num_of_views) ]
        else:
            self.rPhi = [self.add_weight(shape=(self.views_features_dimension[s], self.views_features_dimension[s]),
                                         initializer=tf.initializers.Identity()) for s in range(self.num_of_views) ]

        # Encoders: Conditional gaussian distribution with MLP
        self.cdgn_encoders = [Cdgn_Mlp(
                hidden_layer_size=self.hidden_layer,
                num_of_layer=self.num_of_layer,
                output_dim=self.latent_dim,
                l2_reg=self.l2_reg,
                keep_prob=self.keep_prob,
                use_common_hidden_layer=self.encoder_use_common_hidden_layer
            )
            for s in range(self.num_of_views)]
        
        #Conditional gaussian distribution with MLP
        self.cdgn_encoders = [Cdgn_Mlp(hidden_layer_size=self.hidden_layer, num_of_layer=self.num_of_layer, output_dim=self.latent_dim,
                l2_reg=self.l2_reg, keep_prob=self.keep_prob, use_common_hidden_layer=self.encoder_use_common_hidden_layer)
            for s in range(self.num_of_views)]

    def call(self, views):
        # views_id = list(np.arange(self.num_of_views))
        mean, var, logvar = self.average_encoders(views, views_id = list(np.arange(self.num_of_views))) # mean [batch, hidden_layer]
        view_specific_vars = {}
        for s in range(self.num_of_views):
            mean_s, var_s, logvar_s = self.average_encoders(views, views_id=[s]) # mean_s, [batch, hidden_layer]
            view_specific_vars[s] = {"mean": mean_s, "var": var_s, "logvar": logvar_s}

        #Kullback-Leibler 
        kl_qp =  self.latent_dim + tf.reduce_sum(logvar, axis=1) - tf.reduce_sum(var, axis=1) - tf.reduce_sum( mean ** 2, axis = 1 )
        kl_qp = - 0.5 * kl_qp
        #Cross Entropy Data 
        ce_qp_sum = 0
        for i in range(self.num_of_z_sampled):
            z_sample = mean + tf.multiply(tf.math.sqrt(var), tf.random.normal(tf.shape(mean)))
            # views_sample = [ ( self.cdgn_decoders_mean[s](z_sample)) for s in range(self.num_of_views)] # decoder, TODO    
            views_sample = []
            for s in range(self.num_of_views):
                view_specific_sample = view_specific_vars[s]["mean"] + \
                    tf.multiply(tf.math.sqrt(view_specific_vars[s]["var"]), tf.random.normal(tf.shape(view_specific_vars[s]["mean"])))
                z_sample_s = tf.concat([z_sample, view_specific_sample], axis=1) # z_sample_s, [batch, hidden_layer*2]
                views_sample.append(self.cdgn_decoders_mean[s](z_sample_s))
            if self.decoder_scalar_std :
                n_dist2 = [tfp.distributions.MultivariateNormalTriL(loc=views[s], 
                                            scale_tril=(tf.nn.relu(self.rPhi[s])+1e-6)*tf.eye(self.views_features_dimension[s])[np.newaxis,:]) 
                           for s in range(self.num_of_views)]
            else:
                n_dist2 = [tfp.distributions.MultivariateNormalTriL(loc=views[s], 
                                            scale_tril=(self.rPhi[s][np.newaxis,:]) + 1e-6*tf.eye(self.rPhi[s].shape[0])) 
                           for s in range(self.num_of_views)]
            ce_qp = 0
            for s in range(self.num_of_views) : 
                ce_qp += tf.reduce_mean(n_dist2[s].log_prob(views_sample[s]))
            ce_qp_sum += ce_qp
        ce_qp = ce_qp_sum / self.num_of_z_sampled
        return tf.reduce_mean(kl_qp, axis = 0 ), -ce_qp  # qp and ce
    
    # Encodes views from a limited numbers of views (the same for all instances) or from all views
    def average_encoders(self, views, views_id = []):
        """
        encoder: views: data for each view; w: adj matrix for each view
        """
        if views_id == [] :
            views_id = np.arange(self.num_of_views)
        
        packs = [self.cdgn_encoders[s](views[s:s+1]) for s in views_id] # 
        means = [pack[0] for pack in packs]                         # pack[0] = mu
        logvars = [pack[1] for pack in packs]                       # pack[1] = log var
        vars_ = [tf.math.exp(logvar) for logvar in logvars]         # var
        inv_vars = [1/var for var in vars_]                         # 1/var
        meanXinv_var  =  [ mean * inv_var for mean, inv_var in zip(means,inv_vars)] # mu/var
        mean = 0  # mean = sum(mu/var)
        var = 0   # var = sum(1/var)
        for k in range(len(views_id)):
            mean += meanXinv_var[k] 
            var += inv_vars[k]
        mean = mean / var
        var = 1/ var
        logvar = tf.math.log(var)
        return mean, var, logvar

    # Reconstruct views from a subset of views (the same for all instances)                  
    def get_reconstruct_views_from_someviews(self, views, w, views_id = []):
        mean, var, logvar = self.average_encoders(views, w, views_id = views_id)
        z_sample = mean + tf.multiply( tf.math.sqrt(var), tf.random.normal(mean.shape))
        views_sample = [ ( self.cdgn_decoders_mean[s](z_sample)).numpy() for s in range(self.num_of_views) ]    
        return views_sample 

    # Get all latent space [Z,Z1,...,ZM] from all views
    def get_latents_space(self,views):
        all_latent_space = [self.average_encoders(views, views_id = [id])[0].numpy() for id in range(self.num_of_views)] # Z1,,,,ZM
        all_latent_space.insert(0, self.average_encoders(views)[0].numpy()) # 0: average, Z, all views
        return all_latent_space

    # Get common latent space Z from a subset of views (the same for all instances)     
    def get_common_latent_space_from_someviews(self,views,w,views_id):
        return self.average_encoders(views, w, views_id )[0].numpy()

    # Encode views from a limited numbers of views (can be not the same for all instance) or from all views
    def average_encoders2(self, views, wtab, views_id_tab ):
        views_id0 = np.arange(self.num_of_views)
        if "krylov" in self.encoder_nn_type:
            packs = [self.cdgn_encoders[s]([ views[s:s+1] , wtab[s:s+1] ]) for s in views_id0]
        else:
            packs = [self.cdgns[s](views[s:s+1]) for s in views_id0] 
        means = [pack[0] for pack in packs]
        logvars = [pack[1] for pack in packs]
        vars_ = [tf.math.exp(logvar) for logvar in logvars]
        inv_vars = [1/var for var in vars_]
        existing_views = np.transpose([ np.array([int(id_ in tab) for tab in views_id_tab]) for id_ in range(len(views)) ])
        meanXinv_var  =  [ mean * inv_var for mean, inv_var in zip(means,inv_vars)]
        mean = 0
        var = 0
        for k in range(self.num_of_views):
            mean += existing_views[:,k:k+1]*meanXinv_var[k] 
            var += existing_views[:,k:k+1]*inv_vars[k]
        mean = mean / var
        var = 1/ var
        logvar = tf.math.log(var)
        return mean, var, logvar

    # Get all latent space Z from a subset of views (not the same for all instance) or from all views  
    def get_common_latent_space_from_someviews2(self,views,wtab,views_id):
        return self.average_encoders2(views, wtab, views_id )[0].numpy()
