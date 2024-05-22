"""

This script is the final model
NOV. 17 2022 Marked
"""
from json import load
import os
from re import T  
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import tensorflow as tf
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from utils import process_evaluation
from models.mvae_poe import MVAE_PoE
from models.mvae_poe_private import MVAE_PoE_Private
from sklearn.model_selection import ParameterGrid

from utils.utils import create_views_batch_size, build_graph, load_data
from utils.utils import str2bool
from utils.plot_func import plot_points, plot_relative_error, plot_residual_vs_gt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import json
import joblib


class MVGCCA_U_Trainer(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_train, self.data_test, self.low, self.high = load_data(cfg)
        self.build_model()

    def get_mvgcca_latents_space(self, X):
        """ Return a list containing the common latent space Z and all the views latent space Z_m.
            - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m; M number of views   
            - W : np.array(n x n) weighted adjacency matrix
        """
        W = [np.ones((x.shape[0], x.shape[0])) for x in X]
        batch_views, _, _, _ = create_views_batch_size(X, W, batch_size=X[0].shape[0], shuffle=False)
        Z_list = self.model.get_latents_space(batch_views[0])
        return Z_list


    def build_model(self):
        if self.cfg["model"] == "MVAE_PoE":
            model_mvae = MVAE_PoE(latent_dim=self.cfg['latent_dim'],
                                num_of_layer=self.cfg['num_of_layer'],
                                hidden_layer=self.cfg['hidden_layer'],
                                keep_prob=self.cfg['keep_prob'],
                                encoder_use_common_hidden_layer=self.cfg['encoder_use_common_hidden_layer'],
                                decoder_scalar_std=self.cfg['decoder_scalar_std'],
                                views_dropout_max=self.cfg['views_dropout_max'])
            self.model = model_mvae
        elif self.cfg["model"] == "MVAE_PoE_Private":
            model_mvae = MVAE_PoE_Private(latent_dim=self.cfg['latent_dim'],
                                num_of_layer=self.cfg['num_of_layer'],
                                hidden_layer=self.cfg['hidden_layer'],
                                keep_prob=self.cfg['keep_prob'],
                                encoder_use_common_hidden_layer=self.cfg['encoder_use_common_hidden_layer'],
                                decoder_scalar_std=self.cfg['decoder_scalar_std'])
            self.model = model_mvae

        self.checkpoint = tf.train.Checkpoint(self.model)

    def train(self):
        # Training parameters
        bs = "Y" if self.cfg['bs'] else "N"
        pgs = "Y" if self.cfg['pgs'] else "N"
        ld = "Y" if self.cfg['ld'] else "N"
        exp_name = os.path.join(self.cfg['exp'], f"{self.cfg['pheno_type']}_B{bs}_P{pgs}_L{ld}_{datetime.utcnow().strftime('%B_%d_%Y_%Hh%Mm%Ss')}")
        global_mape = np.inf

        if not os.path.isdir(os.path.join(self.cfg['root_dir'], exp_name)):
            os.makedirs(os.path.join(self.cfg['root_dir'], exp_name))
        
        with open(os.path.join(self.cfg['root_dir'], exp_name, 'config.json'), 'w') as fp:
            json.dump(self.cfg, fp, indent=4)

        learning_rate = self.cfg['learning_rate']
        decay_learning_rate = self.cfg['decay_learning_rate']
        num_of_epochs = self.cfg['num_of_epochs']
        batch_size = self.cfg['batch_size']
        
        target = open(os.path.join(self.cfg['root_dir'], exp_name, "evaluation_te.csv"), "w")
        target.write("epoch,method,r2,mae,rmse,mape\n")

        for e in tqdm(range(num_of_epochs), unit="epochs", disable=False):
            print( "epochs : " + str(e) + "/" + str(num_of_epochs))
            batch_views, batch_adj, batch_s, batch_labels = create_views_batch_size(self.data_train['X'], 
                                                        [np.ones((x.shape[0], x.shape[0])) for x in self.data_train['X']], 
                                                        self.data_train['labels'], batch_size=batch_size, shuffle=True)
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            acc_loss_kl, acc_loss_ce = 0, 0 # ELBO component -> kl : kullbackleibler , ce : cross entropy (features + graph)
            for views, adj, s, y_true in zip(batch_views, batch_adj, batch_s, batch_labels):
                with tf.GradientTape() as tape:
                    loss_kl, loss_ce = self.model(views)
                    loss = loss_kl + loss_ce

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)

                acc_loss_kl += loss_kl
                acc_loss_ce += loss_ce

            if decay_learning_rate :
                optimizer.learning_rate = learning_rate * np.math.pow(1.1, - 50.*(e / num_of_epochs))

            avg_loss = ( acc_loss_kl + acc_loss_ce ) / len(batch_views)
            print( "avg_loss: " + str( avg_loss.numpy()))
            if e % self.cfg['evaluation_interval'] == 0 or e == num_of_epochs-1:
                epoch_save_path = os.path.join(self.cfg['root_dir'], exp_name, "%04d" % e)
                if not os.path.isdir(epoch_save_path):
                    os.makedirs(epoch_save_path)

                # evaluating model performance
                Xtrain, Ytrain = self.data_train["X"], self.data_train["labels"]
                Ztrain_list = self.get_mvgcca_latents_space(Xtrain)
                Ztrain = Ztrain_list[0]
                
                Xtest, Ytest = self.data_test['X'], self.data_test['labels']
                Ztest_list = self.get_mvgcca_latents_space(Xtest)
                Ztest = Ztest_list[0]

                # print("Parameters: " +str(parameters_)) # Z_train, Z_train_labels, Z_test, Z_test_labels, rng
                eva_results_te, linear_reg_model = process_evaluation.evaluate_regression_on_train_test(Ztrain, Ytrain, Ztest, Ytest, (self.low, self.high),
                        save_path=os.path.join(self.cfg['root_dir'], exp_name), epoch=e)

                for k in eva_results_te.keys():
                    target.write(f"{e},{k},{eva_results_te[k]['r2']},{eva_results_te[k]['mae']},{eva_results_te[k]['rmse']},{eva_results_te[k]['mape']}\n")
                    target.flush()
                    print(f"[x] epoch {e}, method = {k}, {eva_results_te[k]}")
                
                if eva_results_te[k]['mape'] < global_mape:
                    global_mape = eva_results_te[k]['mape']
                    # save model
                    # self.model.save(os.path.join(self.cfg['root_dir'], exp_name, "model"), save_format='tf') # correct
                    # self.model = tf.keras.models.load_model(os.path.join(self.cfg['root_dir'], exp_name, "model")) # throw error
                    if not os.path.isdir(os.path.join(self.cfg['root_dir'], exp_name, "model")):
                        os.makedirs(os.path.join(self.cfg['root_dir'], exp_name, "model"))

                    save_path = self.checkpoint.save(os.path.join(self.cfg['root_dir'], exp_name, "model", "model"))
                    joblib.dump(linear_reg_model, os.path.join(self.cfg['root_dir'], exp_name, "model", "reg.joblib")) 
                    print(f"[x] better model obtained, saved to {save_path}")

        target.flush()
        target.close()

        df = pd.read_csv(os.path.join(self.cfg['root_dir'], exp_name, "evaluation_te.csv"))
        min_row = df['mape'].argmin()
        r2 = df['r2'][min_row]
        mae = df['mae'][min_row]
        rmse = df['rmse'][min_row]
        mape = df['mape'][min_row]
        method = df['method'][min_row]
        best_performane_in_test = {"method": method, "r2": r2, "mae": mae, "rmse": rmse, "mape": mape}

        return best_performane_in_test, exp_name


    def restore_and_test(self, exp_name, model_name):
        num_of_views = 0
        if self.cfg['bs']: num_of_views += 1
        if self.cfg['pgs']: num_of_views += 1
        if self.cfg['ld']: num_of_views += 1
        
        self.model.num_of_views = num_of_views
        input_shape = [x.shape for x in self.data_train['X']]
        self.model.build(input_shape) # TODO

        # restore
        # self.model.load_model(os.path.join(exp_name, "model"))
        print(f"[x] restore model from {os.path.join(exp_name, model_name)}")
        # self.model = tf.keras.models.load_model(os.path.join(exp_name, "model"))

        self.checkpoint.restore(os.path.join(exp_name, model_name))
        linear_reg_model = joblib.load(os.path.join(exp_name, "reg.joblib"))

        # load data and predict latent space
        Xtrain, Ytrain = self.data_train["X"], self.data_train["labels"]
        Ztrain_list = self.get_mvgcca_latents_space(Xtrain)
        Ztrain = Ztrain_list[0]
        
        Xtest, Ytest = self.data_test['X'], self.data_test['labels']
        Ztest_list = self.get_mvgcca_latents_space(Xtest)
        Ztest = Ztest_list[0]

        # actual prediction
        Ztest = np.asarray(Ztest)
        Ytrain = np.asarray(Ytrain)
        Ytest = np.array(Ytest)
        scaler = StandardScaler()
        scaler.fit(Ztrain)
        Z_train_transformed = scaler.transform(Ztrain)
        Z_test_transfored = scaler.transform(Ztest)

        Y_test_transformed = Ytest*(self.high-self.low) + self.low
        Y_pred = linear_reg_model.predict(Z_test_transfored)
        Y_pred_transformed = Y_pred*(self.high-self.low) + self.low
        Y_pred_transformed = np.clip(Y_pred_transformed, self.low, self.high)

        r2 = r2_score(Y_test_transformed, Y_pred_transformed)
        mae = mean_absolute_error(Y_test_transformed, Y_pred_transformed)
        rmse = mean_squared_error(Y_test_transformed, Y_pred_transformed, squared=False)
        mape = mean_absolute_percentage_error(Y_test_transformed, Y_pred_transformed)
        print(f"[x] test performance: r2 = {r2}, mae = {mae}, rmse = {rmse}, mape = {mape}")

        self.plot(Y_test_transformed, Y_pred_transformed, os.path.join(exp_name, "best"))

        self.linear_reg_model = linear_reg_model

        return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape, 'gt': Y_test_transformed, 'pred': Y_pred_transformed}

    def plot(self, gt, pred, path):
        interval = 5000 if np.max(gt) - np.min(gt) > 10000 else 1000
        plot_points(gt, pred, self.cfg['pheno_type'], path+"point.png", interval)
        plot_relative_error(gt, pred, self.cfg['pheno_type'], path+"re.png")
        plot_residual_vs_gt(gt, pred, self.cfg['pheno_type'], path+"residual.png", interval, interval)


def is_exist(target_csv_path, params):
    df = pd.read_csv(target_csv_path)
    if df[(df['bs']==params['bs']) & (df['pgs']==params['pgs']) & (df['ld']==params['ld']) & 
          (df['num_of_layer']==params['num_of_layer']) & (df['latent_dim']==params['latent_dim']) & 
          (df['hidden_layer']==params['hidden_layer']) & (df['model']==params['model'])].shape[0]>0:
          return True
    else:
        return False

if __name__ == '__main__':
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="meta_data")
    parser.add_argument('--exp', type=str, default="exp_meta")

    # model
    parser.add_argument('--model', type=str, default="MVAE_PoE")
    parser.add_argument('--num_of_epochs', type=int, default=1001, help='Number of epochs.')
    parser.add_argument('--device', default='0')
    parser.add_argument('--decoder_scalar_std', type=str2bool, default=True, help='True or False. Decide whether or not to use scalar matrix as covariance matrix for gaussian decoder.')
    parser.add_argument('--encoder_nn_type', default='mlp', help='Encoders neural networks. Only Mlp is available.')
    parser.add_argument('--encoder_use_common_hidden_layer', type=str2bool, default=False, help='True or False. Whether or not to use differents hidden layers to calculate the mean and variance of encoders')
    parser.add_argument('--num_of_layer', type=int, default=2, help='Number of layer for encoders AND decoders.')
    parser.add_argument('--hidden_layer', type=int, default=256, help='Size of hidden layers for encoders AND decoders.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--decay_learning_rate', type=str2bool, default = True, help='True or False. Apply or not a decay learning rate.')
    parser.add_argument('--batch_size', type=int, default=-1, help='Batch size.')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='Dropout rate applied to every hidden layers.')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent space.')
    
    # evaluation
    parser.add_argument('--evaluation_interval', type=int, default=100)

    # DATA
    parser.add_argument('--bs', type=str2bool, default=True)
    parser.add_argument('--pgs', type=str2bool, default=True)
    parser.add_argument('--ld', type=str2bool, default=True)
    parser.add_argument('--data_root', type=str, default="./data")
    parser.add_argument('--mapping_df_path', type=str, default="data/mapping.csv")
    parser.add_argument('--pheno_type', type=str, default="424")
    parser.add_argument('--train_ratio', type=float, default=0.8)

    # grid search
    parser.add_argument('--grid_search', type=str2bool, default=True)

    parser.add_argument('--train', type=str2bool, default=True)
    # parser.add_argument('--exp_name', type=str, default="exp4/Nonlinear_Fall_GY_DY_CY_October_04_2022_20h42m47s/model")
    # parser.add_argument('--model_name', type=str, default="model-4")

    args = parser.parse_args()


    if args.train:
        device = '/cpu:0' if args.device == '-1' or args.device == '' else '/gpu:'+args.device
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)

        # data, rng = process_data.load_u19(parameters)
        # data_train, data_test = process_data.split_data_intwo(data, cut=args.train_ratio, shuffle=False)

        parameters = {}
        parameters['model'] = [args.model]
        parameters['root_dir'] = [args.root_dir]
        parameters['exp'] = [args.exp]
        parameters['num_of_epochs'] = [args.num_of_epochs]
        parameters['device'] = [args.device]
        parameters['decoder_scalar_std'] = [args.decoder_scalar_std]
        parameters['encoder_nn_type'] = [args.encoder_nn_type]
        parameters['encoder_use_common_hidden_layer'] = [args.encoder_use_common_hidden_layer]
        parameters['num_of_layer'] = [args.num_of_layer]                    #
        parameters['hidden_layer'] = [args.hidden_layer]                    #
        parameters['learning_rate'] = [args.learning_rate]
        parameters['decay_learning_rate'] = [args.decay_learning_rate]
        parameters['batch_size'] = [args.batch_size]
        parameters['keep_prob'] = [args.keep_prob]
        parameters['views_dropout_max'] = [args.views_dropout_max]
        parameters['latent_dim'] = [args.latent_dim]                        #
        parameters['evaluation_interval'] = [args.evaluation_interval]      
        parameters['bs'] = [args.bs]                                      #
        parameters['pgs'] = [args.pgs]                                      #
        parameters['ld'] = [args.ld]                                  #
        parameters['data_root'] = [args.data_root]
        parameters['mapping_df_path'] = [args.mapping_df_path]
        parameters['pheno_type'] = [args.pheno_type]
        parameters['train_ratio'] = [args.train_ratio]

        if args.grid_search:
            #parameters['model'] = ["MVAE_PoE", "MVAE_PoE_Private"]
            parameters['model'] = ["MVAE_PoE"]
            parameters['bs'] = [True, False]
            parameters['pgs'] = [True, False]
            parameters['ld'] = [True, False]
            parameters['latent_dim'] = [128]
            parameters["encoder_nn_type"] = ["MLP"]
            parameters["num_of_layer"] = [2]
            parameters["hidden_layer"] = [128]#[32, 64, 96, 128, 256]
            parameters["decoder_scalar_std"] = [True]
            # parameters["pheno_type"] = ["Nonlinear_Stance"]
            
        print(f"[x] start grid search, # {len(list(ParameterGrid(parameters)))} experiments")
        if not os.path.isdir(os.path.join(args.root_dir, args.exp)):
            os.makedirs(os.path.join(args.root_dir, args.exp))
        
        if os.path.isfile(os.path.join(args.root_dir, args.exp, f"best_results_{args.pheno_type}.csv")):
            n_exp_have_done = len(open(os.path.join(args.root_dir, args.exp, f"best_results_{args.pheno_type}.csv"), "r").readlines())-1
            target = open(os.path.join(args.root_dir, args.exp, f"best_results_{args.pheno_type}.csv"), "a")
        else:
            n_exp_have_done = 0
            target = open(os.path.join(args.root_dir, args.exp, f"best_results_{args.pheno_type}.csv"), "w")
            target.write("model,bs,pgs,ld,num_of_layer,hidden_layer,latent_dim,r2,rmse,mae,mape,path\n")
            target.flush()

        total = len(list(ParameterGrid(parameters)))
        
        c = 0
        for parameters_, parameters_id in zip(list(ParameterGrid(parameters)), range(len(list(ParameterGrid(parameters))))):
            if is_exist(os.path.join(args.root_dir, args.exp, f"best_results_{args.pheno_type}.csv"), parameters_):
                c += 1
                continue

            n_views = 0
            if parameters_['bs'] == True:
                n_views += 1
            if parameters_['pgs'] == True:
                n_views += 1
            if parameters_['ld'] == True:
                n_views += 1

            if n_views >= 1:
                dicts = []
                print(f"--------------{c}/{total}---------")
                trainer = MVGCCA_U_Trainer(parameters_)
                best_dict, exp_path = trainer.train()
                
                target.write(f"{parameters_['model']},{parameters_['bs']},{parameters_['pgs']},{parameters_['ld']},"\
                            f"{parameters_['num_of_layer']},{parameters_['hidden_layer']},{parameters_['latent_dim']},"\
                            f"{best_dict['r2']},{best_dict['rmse']},{best_dict['mae']},{best_dict['mape']},"\
                            f"{exp_path}\n")
                target.flush()

            c += 1

        target.flush()
        target.close()
    
    else:

        parameters = {}
        parameters['model'] = args.model
        parameters['root_dir'] = args.root_dir
        parameters['exp'] = args.exp
        parameters['num_of_epochs'] = args.num_of_epochs
        parameters['device'] = args.device
        parameters['decoder_scalar_std'] = args.decoder_scalar_std
        parameters['encoder_nn_type'] = args.encoder_nn_type
        parameters['encoder_use_common_hidden_layer'] = args.encoder_use_common_hidden_layer
        parameters['num_of_layer'] = args.num_of_layer                    #
        parameters['hidden_layer'] = args.hidden_layer                    #
        parameters['learning_rate'] = args.learning_rate
        parameters['decay_learning_rate'] = args.decay_learning_rate
        parameters['batch_size'] = args.batch_size
        parameters['keep_prob'] = args.keep_prob
        parameters['views_dropout_max'] = args.views_dropout_max
        parameters['latent_dim'] = args.latent_dim                        #
        parameters['evaluation_interval'] = args.evaluation_interval      
        parameters['bs'] = args.bs                                    #
        parameters['pgs'] = args.pgs                                      #
        parameters['ld'] = args.ld                            #
        parameters['data_root'] = args.data_root
        parameters['pheno_type'] = args.pheno_type
        parameters['train_ratio'] = args.train_ratio

        trainer = MVGCCA_U_Trainer(parameters)
        trainer.restore_and_test(args.exp_name, args.model_name)
