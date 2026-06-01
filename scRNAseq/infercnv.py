#!/usr/bin/env python
# coding: utf-8

# # ULMS infercnvpy for the revision of the paper
# - Uses adata.raw (log normalized data) that is not batch corrected, with all immune cells as a reference

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import infercnvpy as cnv
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpacnv


# In[2]:


# version control
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scanpy:", sc.__version__)
print("seaborn:", sns.__version__)
print("infercnv:", cnv.__version__)


# In[3]:


# Set up input and output directories
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent
print(PROJECT_DIR)

DATA_DIR = PROJECT_DIR / 'objects'
print(DATA_DIR)

output_dir = jpacnv.create_output_dir(PROJECT_DIR, 'cnv', change_dir=True)


# In[4]:


N_JOBS = 16
WINDOW_SIZE = 500


# # Running the infercnvpy pipeline

# In[5]:


# careful with the input directory
adata = sc.read_h5ad(DATA_DIR / 'annotated.h5ad')
adata


# In[6]:


# this is key - need to use all the genes, not just the HVG, for cnv analysis
# adata.raw contains the lognormalized counts prior to HVG selection.
rawdata = adata.raw.to_adata()
ref = PROJECT_DIR / 'ref/GRCh38-2024-A.txt'
sample_adata = sc.read_h5ad(PROJECT_DIR / 'preprocessed/Batch01_pp_singlet_adata.h5ad')
ensg_map = sample_adata.var['gene_ids'].to_dict()
del sample_adata
rawdata = jpacnv.gof_format(rawdata, ref=ref, ensg_map=ensg_map)


# In[7]:


types = rawdata.obs["celltype"].unique()
print(*types)
rawdata.var.loc[:, ["ensg", "chromosome", "start", "end"]].head()


# In[ ]:


# this time keep the X chromosome and exclude only the Y chromosome
# Increased the window size to more genes to make the contrast between tumor and immune cells more obvious and less noisy
immune_types = ["B", "Myeloid", "T_and_NK", "Mast", "Plasma", "pDC"]
cnv.tl.infercnv(rawdata, 
                reference_key="celltype", 
                reference_cat=immune_types,
                exclude_chromosomes=('chrY',), 
                window_size=WINDOW_SIZE, 
                n_jobs=N_JOBS
               ) 


# In[ ]:


rawdata = jpacnv.cnv_plot(rawdata, resolution=1.7)


# In[ ]:


ad_path = Path(output_dir / 'rawdata_with_cnv.h5ad')
rawdata.write_h5ad(ad_path)
del rawdata


# # Making a table of median CNV scores per cluster

# In[ ]:


adata = sc.read_h5ad(output_dir / 'rawdata_with_cnv.h5ad')
adata


# In[ ]:


leiden_key = 'leiden1_7'
median_per_cluster = adata.obs.groupby(leiden_key, observed=True)['cnv_score'].median()
adata.obs['cnv_score_cluster_median'] = adata.obs[leiden_key].map(median_per_cluster)
adata.obs['cnv_score_cluster_median']


# In[ ]:


# Create a bar plot for genes by counts
median_per_cluster = median_per_cluster.reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x=leiden_key, y='cnv_score', data=median_per_cluster)
plt.title('Median CNV score for each cluster')
plt.ylabel('median CNV score per cluster')
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig(f'median_cnv_score_{leiden_key}_barplot.png')
plt.show()


# # Plotting individual clusters on the CNV UMAP

# In[ ]:


adata = sc.read_h5ad(output_dir / 'rawdata_with_cnv.h5ad')
adata


# In[ ]:


leiden_key = 'leiden1_7'
cats = adata.obs[leiden_key].cat.categories.tolist()
print(*cats)


# In[ ]:


for cat in cats:
    target_cluster = cat
    adata.obs['is_target_cluster'] = (adata.obs[leiden_key] == target_cluster)
    cnv.pl.umap(adata, 
                color='is_target_cluster', 
                groups=[False, True], 
                palette=['lightgray', 'red'], 
                size=10, 
                title=f'Highlight Leiden cluster {target_cluster}', 
                save=f'cnv_umap_{target_cluster}.png'
               )


# In[ ]:


celltypes = adata.obs["celltype"].cat.categories.tolist()
print(*celltypes)


# In[ ]:


for celltype in celltypes:
    target_annotation = celltype
    adata.obs['is_target_annotation'] = (adata.obs['celltype'] == target_annotation)
    cnv.pl.umap(adata, 
                color='is_target_annotation', 
                groups=[False, True], 
                palette=['lightgray', 'red'], 
                size=10, 
                title=f'Highlight Leiden cluster {target_annotation}',
                save=f'cnv_umap_{target_annotation}.png'
               )


# In[ ]:


sc.pl.umap(adata, color=['leiden1_7', 'cnv_score'])


# # Plotting cnv score on tumor subset umap

# In[ ]:


# # careful with the input directory
# path = Path('/oak/stanford/groups/longaker/ULMS/analysis_v3/objects/tumor_subset_raw.h5ad')
# adata = sc.read_h5ad(path)
# adata


# In[ ]:


# # careful with the input directory
# path = Path(output_dir / 'rawdata_with_cnv.h5ad')
# cnv = sc.read_h5ad(path)
# cnv


# In[ ]:


# adata.obs['cnv_score'] = cnv.obs.loc[adata.obs.index, 'cnv_score']
# adata.obs


# In[ ]:


# sc.pl.umap(adata, color='cnv_score', save='cnv_score_tumor_subset.png')

