#!/usr/bin/env python
# coding: utf-8

# # ULMS Revision: Preprocessing / quality control of all samples
# - This notebook uses the same thresholds as the original submission.
# - DoubletDetection package for doublet removal

# In[ ]:


import os
import sys
from pathlib import Path
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpasc


# In[ ]:


# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("seaborn:", sns.__version__)
sns.set_theme()


# In[ ]:


CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)
output_dir = jpasc.create_output_dir(PROJECT_DIR, 'qc', change_dir=True)

data_dir = PROJECT_DIR / 'cellbender_outs/Final_outputs'
print(data_dir)


# In[ ]:


# Global settings
MIN_GENES = 100
MAX_MITO = 50
N_ITERS = 20
P_THRESH = 1e-16
VOTER_THRESH = 0.5


# # Original cellbender outputs minus the high-grade cervical sarcoma sample
# - These are preprocessed by batch since one batch (containing 3 samples) was multiplexed with antibodies

# In[ ]:


# Import the old cellbender outputs
adata_list = jpasc.import_and_label_data(data_dir, label='batch', keyword='Batch')
adata_list.sort(key=lambda x: x.uns['filename']) # sorts the list of anndata objects


# In[ ]:


# Checking var names are unique
for adata in adata_list:
    print(adata.uns['batch'])
    if (len(np.unique(adata.var_names)) == len(adata.var_names)): print("Variable names are unique.")
    else: print("false")


# In[ ]:


#debugging
adata_list[0].obs['batch']


# In[ ]:


metadata = pd.read_csv(PROJECT_DIR / 'ref/metadata_hto.csv', dtype={'sample': str}, index_col=0)
metadata.index = metadata.index + "_Batch01"
metadata


# In[ ]:


# The first experiment is hashed, so we need to label the hashed samples.
# Demultiplexing code was done previously in R: see ulms_demux_cb_JA.R

#ulms01 hashes three dates
adata = adata_list[0]
adata.obs['sample'] = metadata.loc[adata.obs.index, 'sample']
adata_list[0] = adata


# In[ ]:


#debugging
adata_list[0].obs


# In[ ]:


sample_map = {
    'Batch02' : 'Sample02',
    'Batch03' : 'Sample01',
    'Batch04' : 'Sample05',
    'Batch05' : 'Sample06',
    'Batch06' : 'Sample07',
    'Batch07' : 'Sample08',
    'Batch08' : 'Sample09',
    'Batch09' : 'Sample10',
    'Batch10' : 'Sample10',
    'Batch11' : 'Sample11',
    'Batch12' : 'Sample12',
    'Batch13' : 'Sample13',
    'Batch14' : 'Sample14',
}

batch01 = adata_list.pop(0)
for adata in adata_list:
    adata.obs['sample'] = adata.obs['batch'].map(sample_map)
adata_list.insert(0, batch01)


# In[ ]:


test_data = adata_list[0].obs['sample']
# Create histogram
sns.histplot(test_data, bins=5, kde=False)

# Add titles and labels
plt.title('Histogram of Samples in Batch01')
plt.xlabel('Sample')
plt.ylabel('Number of cells')

# Show the plot
plt.savefig('batch01_histogram.png')
plt.close()


# In[ ]:


len(adata_list)


# In[ ]:


adata_list


# In[ ]:


print()
print("Quality Control\n")
preprocessed_list = []
for adata in adata_list:
    adata = jpasc.qc_adata(adata, label='batch', species='human')
    adata = jpasc.filter_counts(adata, label='batch', method='standard', min_genes=MIN_GENES)
    adata = jpasc.filter_mt(adata, label='batch', method='standard', max_mito=MAX_MITO)
    adata = jpasc.deciteseq(adata)
    preprocessed_list.append(adata)


# In[ ]:


print()
print("Finding Doublets\n")
singlet_list = preprocessed_list.copy()
# find the doublets
# this construction transfers the doublet and doublet score labels without modifying the original data in any way
# this allows you to keep all the genes and not transform the count matrix
for adata in singlet_list:
    singlet = jpasc.dd_find_doublets(adata, label='batch', algorithm="louvain", n_iters=N_ITERS, p_thresh=P_THRESH, voter_thresh=VOTER_THRESH)
    adata.obs["doublet"] = singlet.obs.loc[adata.obs.index, "doublet"]
    adata.obs["doublet_score"] = singlet.obs.loc[adata.obs.index, "doublet_score"]


# In[ ]:


# filter out the doublets
print()
print("Removing Doublets\n")
for i, adata in enumerate(singlet_list):
    print(adata.uns['batch'])
    n_cells = adata.n_obs
    print(f"Number of cells before removing doublets: {n_cells}")
    adata_singlet = adata[adata.obs['doublet'] == 0.0]
    singlet_list[i] = adata_singlet
    n_singlets = adata_singlet.n_obs
    print(f"Number of cells after removing doublets: {n_singlets}")
    print(f"Filtered out {n_cells - n_singlets} doublets")
    print(f"Expected doublets: {n_cells / 1000 * 0.008 * n_cells}") # From 10x Genomics


# In[ ]:


for adata in singlet_list:
    print(adata.uns['batch'])
    print(np.unique(adata.obs['sample']))
    print(adata.shape)
    print()


# In[ ]:


# save preprocessed anndatas
output_dir = jpasc.create_output_dir(PROJECT_DIR, 'preprocessed', change_dir=True)
for adata in singlet_list:
    batch = adata.uns['batch']
    file_path = os.path.join(output_dir, f'{batch}_pp_singlet_adata.h5ad')
    adata.write(file_path)
print(f"Saved {len(singlet_list)} anndatas to {output_dir}")


# In[ ]:


# Clean up
del adata_list
del preprocessed_list
del singlet_list
del adata
del batch


# # New anndatas added for the revision

# In[ ]:


output_dir = jpasc.create_output_dir(PROJECT_DIR, 'qc', change_dir=True)


# In[ ]:


adata_list = jpasc.import_and_label_data(data_dir, label='sample', keyword='Sample')
adata_list.sort(key=lambda x: x.uns['filename']) # sorts the list of anndata objects


# In[ ]:


# Checking var names are unique
for adata in adata_list:
    print(adata.uns['sample'])
    if (len(np.unique(adata.var_names)) == len(adata.var_names)): print("Variable names are unique.")
    else: print("false")


# In[ ]:


# Add in the batch information
batch_map = {
    'Sample15' : 'Batch15',
    'Sample16' : 'Batch16',
    'Sample17' : 'Batch17',
    'Sample18' : 'Batch18',
    'Sample19' : 'Batch19',
    'Sample20' : 'Batch20',
    'Sample21' : 'Batch19',
    'Sample22' : 'Batch21',
    'Sample23' : 'Batch22', 
}
for adata in adata_list:
    adata.obs['batch'] = adata.obs['sample'].map(batch_map)


# In[ ]:


len(adata_list)


# In[ ]:


adata_list


# In[ ]:


preprocessed_list = []
for adata in adata_list:
    adata = jpasc.qc_adata(adata, label='sample', species='human')
    adata = jpasc.filter_counts(adata, label='sample', method='standard', min_genes=MIN_GENES)
    adata = jpasc.filter_mt(adata, label='sample', method='standard', max_mito=MAX_MITO)
    adata = jpasc.deciteseq(adata)
    preprocessed_list.append(adata)


# In[ ]:


print()
print("Finding Doublets\n")
singlet_list = preprocessed_list.copy()
# find the doublets
# this construction transfers the doublet and doublet score labels without modifying the original data in any way
# this allows you to keep all the genes and not transform the count matrix
for adata in singlet_list:
    singlet = jpasc.dd_find_doublets(adata, label='sample', algorithm="louvain", n_iters=N_ITERS, p_thresh=P_THRESH, voter_thresh=VOTER_THRESH)
    adata.obs["doublet"] = singlet.obs.loc[adata.obs.index, "doublet"]
    adata.obs["doublet_score"] = singlet.obs.loc[adata.obs.index, "doublet_score"]


# In[ ]:


# filter out the doublets
print()
print("Removing Doublets\n")
for i, adata in enumerate(singlet_list):
    print(adata.uns['sample'])
    n_cells = adata.n_obs
    print(f"Number of cells before removing doublets: {n_cells}")
    adata_singlet = adata[adata.obs['doublet'] == 0.0]
    singlet_list[i] = adata_singlet
    n_singlets = adata_singlet.n_obs
    print(f"Number of cells after removing doublets: {n_singlets}")
    print(f"Filtered out {n_cells - n_singlets} doublets")
    print(f"Expected doublets: {n_cells / 1000 * 0.008 * n_cells}") # From 10x Genomics


# In[ ]:


for adata in singlet_list:
    print(adata.uns['sample'])
    print(np.unique(adata.obs['batch']))
    print(adata.shape)
    print()


# In[ ]:


# save preprocessed anndatas
output_dir = jpasc.create_output_dir(PROJECT_DIR, 'preprocessed', change_dir=True)
for adata in singlet_list:
    sample = adata.uns['sample']
    file_path = os.path.join(output_dir, f'{sample}_pp_singlet_adata.h5ad')
    adata.write(file_path)
print(f"Saved {len(singlet_list)} anndatas to {output_dir}")

