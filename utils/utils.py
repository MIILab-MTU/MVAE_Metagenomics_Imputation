import numpy as np
import pandas as pd
import os

def str2bool(string):
    """ Convert string to corresponding boolean.
        -  string : str
    """
    if string in ["True","true","1"]:
        return True
    elif string in ["False","false","0"]:
        return False
    else :
        return False
    

def split_and_norm(data, ratio):
    assert len(data.shape) == 2
    data0, data1 = data[:int(data.shape[0]*ratio)], data[int(data.shape[0]*ratio):]
    for i in range(data0.shape[1]):
        l, h = np.min(data0[:, i]), np.max(data0[:, i])
        data0[:, i] = (data0[:, i] - l) / (h - l)
        data1[:, i] = (data1[:, i] - l) / (h - l)
    return data0, data1


def load_data_within(args, remove_outliers=False, outlier_threshold=0.9):
    root_path = args['data_root']
    label_column = args['pheno_type']
    train_ratio = args['train_ratio']
    mapping_df = pd.read_csv(args['mapping_df_path'])

    label_df = pd.read_csv(os.path.join(root_path, f"meta.csv"))
    labels = label_df[str(label_column)].values

    if remove_outliers:
        mean, std = np.mean(labels), np.std(labels)
        selected_indices = np.where(labels < mean + outlier_threshold * mean)[0]
    else:
        selected_indices = np.arange(labels.shape[0])
    
    labels = labels[selected_indices] # select labels

    labels_tr = labels[:int(labels.shape[0]*train_ratio)]
    labels_te = labels[int(labels.shape[0]*train_ratio):]

    low, high = np.min(labels_tr), np.max(labels_tr)
    labels_tr = (labels_tr - low)/(high - low)
    labels_te = (labels_te - low)/(high - low)

    inputs = []
    reference_ids = np.unique(mapping_df['reference_id'])
    for reference_id in reference_ids:
        data_x = label_df[str(reference_id)].values[selected_indices]
        inputs.append(data_x)
    inputs = np.stack(inputs)
    inputs = inputs.T

    inputs_tr, inputs_te = split_and_norm(inputs, train_ratio)
    return inputs_tr, inputs_te, labels_tr, labels_te, low, high


def load_data_within_nearest(args, remove_outliers=False, outlier_threshold=0.9):
    root_path = args['data_root']
    label_column = args['pheno_type']
    train_ratio = args['train_ratio']
    mapping_df = pd.read_csv(args['mapping_df_path'])

    label_df = pd.read_csv(os.path.join(root_path, f"meta.csv"))
    labels = label_df[str(label_column)].values

    if remove_outliers:
        mean, std = np.mean(labels), np.std(labels)
        selected_indices = np.where(labels < mean + outlier_threshold * mean)[0]
    else:
        selected_indices = np.arange(labels.shape[0])
    
    labels = labels[selected_indices] # select labels

    labels_tr = labels[:int(labels.shape[0]*train_ratio)]
    labels_te = labels[int(labels.shape[0]*train_ratio):]

    low, high = np.min(labels_tr), np.max(labels_tr)
    labels_tr = (labels_tr - low)/(high - low)
    labels_te = (labels_te - low)/(high - low)

    reference_id = str(mapping_df[mapping_df['meta_id']==int(label_column)]['reference_id'].values[0])
    df_data = pd.read_csv(os.path.join(root_path, f"bs_{reference_id}.csv"))
    df_data = df_data[[col for col in df_data.columns if col not in ['MM_ID', 'LOS_ID', 'Unnamed: 0']]]
    df_data = df_data.fillna(0.)
    data = df_data.to_numpy()
    data = data[selected_indices]
    inputs_tr, inputs_te = split_and_norm(data, train_ratio)
    return inputs_tr, inputs_te, labels_tr, labels_te, low, high


def load_data(args, remove_outliers=False, outlier_threshold=0.9):

    root_path = args['data_root']
    label_column = args['pheno_type']
    train_ratio = args['train_ratio']
    mapping_df = pd.read_csv(args['mapping_df_path'])

    label_df = pd.read_csv(os.path.join(root_path, f"meta.csv"))
    labels = label_df[str(label_column)].values

    if remove_outliers:
        mean, std = np.mean(labels), np.std(labels)

        # selected_indices = np.where(np.logical_and(labels < np.quantile(labels, outlier_threshold), 
        #                                            labels > np.quantile(labels, 1 - outlier_threshold)))[0]
        #selected_indices = np.where(np.logical_and(labels < mean + outlier_threshold * std, labels > mean - outlier_threshold * std))[0]
        # selected_indices = np.where(np.logical_and(labels < mean + outlier_threshold * mean, 
        #                                            labels > mean - outlier_threshold * std))[0]
        selected_indices = np.where(labels < mean + outlier_threshold * mean)[0]
    else:
        selected_indices = np.arange(labels.shape[0])
    
    labels = labels[selected_indices] # select labels

    labels_tr = labels[:int(labels.shape[0]*train_ratio)]
    labels_te = labels[int(labels.shape[0]*train_ratio):]

    low, high = np.min(labels_tr), np.max(labels_tr)
    labels_tr = (labels_tr - low)/(high - low)
    labels_te = (labels_te - low)/(high - low)
    
    tr_x = []
    te_x = []
    n_views = 0
    if args['bs']:
        n_views += 1

        reference_id = str(mapping_df[mapping_df['meta_id']==int(label_column)]['reference_id'].values[0])
        corr = mapping_df[mapping_df['meta_id']==int(label_column)]['corr'].values[0]
        df_data = pd.read_csv(os.path.join(root_path, f"bs_{reference_id}.csv"))

        df_data = df_data[[col for col in df_data.columns if col not in ['MM_ID', 'LOS_ID', 'Unnamed: 0']]]
        df_data = df_data.fillna(0.)
        data = df_data.to_numpy()
        data = data[selected_indices]
        
        data0, data1 = split_and_norm(data, train_ratio)
        tr_x.append(data0)
        te_x.append(data1)
        print(f"[x] loading burden scores, shape = {data.shape}")
    
    if args['pgs']:
        n_views += 1
        df_data = pd.read_csv(os.path.join(root_path, f"pgs.csv"))
        df_data = df_data[[col for col in df_data.columns if col not in ['MM_ID', 'LOS_ID']]]
        df_data = df_data.fillna(0.)
        data = df_data.to_numpy()
        data = data[selected_indices]

        data0, data1 = split_and_norm(data, train_ratio)
        tr_x.append(data0)
        te_x.append(data1)
        print(f"[x] loading pgs scores, shape = {data.shape}")
    
    if args['ld']:
        n_views += 1
        df_data = pd.read_csv(os.path.join(root_path, f"ld.csv"))
        df_data = df_data[[col for col in df_data.columns if col not in ['MM_ID', 'LOS_ID']]]
        df_data = df_data.fillna(0.)
        data = df_data.to_numpy()

        data = data[selected_indices]
        data0, data1 = split_and_norm(data, train_ratio)
        tr_x.append(data0)
        te_x.append(data1)
        print(f"[x] loading pgs scores, shape = {data.shape}")

    data_tr = {'X': tr_x, 'labels': labels_tr}
    data_te = {'X': te_x, 'labels': labels_te}
    return data_tr, data_te, low, high, corr



def create_views_batch_size(X, labels=[], batch_size=32, shuffle=True):
    """ Creates a list of batch of multiviews data. 
        - X : [np.array(n x d1),...,np.array(n x dM)] multivews features ; n number of instances; dm dimension of views m ; M number of views   
    """
    # training data for each view
    views = [np.copy(original_view) for original_view in X]
    labels = np.array(labels)
    labels = np.copy(labels)

    n = views[0].shape[0]
    s = np.arange(n) 
    if shuffle:
        np.random.shuffle(s)
        for i, view in enumerate(views) :
            views[i] = view[s]
        labels = labels[s]
    
    if batch_size == -1:
        batch_indice = [[0, n]]
    else:
        q = n//batch_size
        if n%batch_size == 0:
            batch_indice = [[k*batch_size,(k+1)*batch_size] for k in range(q)]
        else:
            batch_indice = [[k*batch_size,(k+1)*batch_size] for k in range(q)] + [[q*batch_size,n]]
    batch_views = [[view[ind[0]:ind[1]] for view in views ] for ind in batch_indice]

    # retrive graph adj matrixs
    batch_s = [s[ind[0]:ind[1]] for ind in batch_indice]
    batch_labels = []
    for ind in batch_indice:
        batch_labels.append(labels[ind[0]:ind[1]])
    return batch_views, batch_s, batch_labels


