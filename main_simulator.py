from ST_simulator.simulators.simulator_naive import sim_naive_spot,sim_naive_spot_splatter

import scanpy as sc
import os
import numpy as np
import gc

data_folder_name = 'sc_heart'

real_adata = sc.read(f'./real_data/{data_folder_name}/Ischemia-snRNA-Spatial multi-omic map of human myocardial infarction.h5ad')
real_adata.X = real_adata.raw.X.copy().toarray()
real_adata.var_names_make_unique()

print(real_adata)

# Generate n random numbers
np.random.seed(123)
n = 20
seeds = np.random.randint(200, size=n)

for seed in seeds[:8]:
    print(f'Simulating ST for Seed: {seed}')
    save_path = f'./output/{data_folder_name}/simulation_results/{seed}/'
    os.makedirs(save_path, exist_ok=True)
    RCTD_spot_sim = sim_naive_spot(use_real_adata=real_adata, level='cell', spot_diameter=10,
                                   ctkey='cell_type_original', method='RCTD', file_path=save_path, seed = seed, Min=1, Max=20)

    print(RCTD_spot_sim)
    del RCTD_spot_sim
    gc.collect()

del real_adata
gc.collect()

# real_adata = sc.read("../real_data/STARmap_15ct/15ct_realdata.h5ad")

# print(real_adata.X)

# save_path = '../output/'

# RCTD_spot_sim =sim_naive_spot(use_real_adata=real_adata, level='spot', spot_diameter=500,
#                    ctkey='celltype', method='RCTD', file_path=save_path, seed = 123)
# print(RCTD_spot_sim)

# del RCTD_spot_sim
# gc.collect()

# real_adata = sc.read("../real_data/sc_heart/Ischemia-snRNA-Spatial multi-omic map of human myocardial infarction.h5ad")

# print(real_adata)
# real_adata.X = real_adata.raw.X.copy().toarray()

# save_path = '../output/'

# RCTD_spot_sim =sim_naive_spot(use_real_adata=real_adata, level='cell', spot_diameter=10,
#                    ctkey='cell_type_original', method='RCTD', file_path=save_path, seed = 123, Min=1, Max=10)
# print(RCTD_spot_sim)

# del RCTD_spot_sim
# gc.collect()

# sim_data = sc.read("../output/RCTD_simu_cell_level.h5ad")

# print(sim_data)
# print(sim_data.var)
# print(sim_data.uns['W'].shape)
# print(sim_data.obsm)

# marker_genes = ['CTDSP2', 'MS4A1', 'CD8A', 'CD8B', 'LYZ']
# # marker_genes = ["Aamp"]

# gene_df = sc.get.obs_df(sim_data, keys=marker_genes, gene_symbols="feature_name")
 
# print(gene_df)

# print(np.sum(sim_data.uns['W'],axis=1))
# print(np.mean(np.sum(sim_data.uns['W'],axis=1)))