import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split

def get_simulation_data(file_path, cell_type_idx , marker_genes):
    adata = sc.read(file_path)
    df_gene_exp = sc.get.obs_df(adata, keys=marker_genes, gene_symbols="feature_name")

    x = df_gene_exp.values

    if type(cell_type_idx) is str:
        cell_type_idx = np.where(adata.uns['celltype_name']==cell_type_idx)[0][0]
    y = adata.uns['W'][:, cell_type_idx].toarray()

    return x, y

def get_all_simulation_data(file_paths, cell_type_idx, marker_genes):
    x_all = []
    y_all = []
    for file_path in file_paths:
        x, y = get_simulation_data(file_path, cell_type_idx, marker_genes)
        x_all.extend(x)
        y_all.extend(y)

    return x_all, y_all

def get_train_and_test_data(x_all, y_all,seed, train_ratio = 0.8):
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, train_size=train_ratio, shuffle=True, random_state=seed)

    return X_train, X_test, y_train, y_test