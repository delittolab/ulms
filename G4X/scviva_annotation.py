#!/usr/bin/env python
# coding: utf-8

# # Manually annotating the scVIVA model of the ULMS G4X dataset
# - annotates the scVIVA_2 clustered object (proseg). The one with the reduced learning rate.
# - This is for the revision of the paper.

# In[ ]:


import os
import sys
import numpy as np
import scanpy as sc
import torch
import scvi
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

module_path = '/labs/delitto/james/functions/'
sys.path.append(module_path)
import jpascvi


# In[ ]:


# version control
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("scvi:", scvi.__version__)

plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
plt.interactive = False
plt.ioff()
sc.settings.autoshow = False
plt.rcParams['savefig.dpi'] = 300
sc.set_figure_params(dpi_save=300, facecolor='white')

sc.settings.n_jobs = -1  # Use all available cores
SEED = 1234
scvi.settings.seed = SEED
torch.set_float32_matmul_precision("high")


# In[ ]:


CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
print(PARENT_DIR)

# Making an output directory using the pathlib package
output_dir = jpascvi.create_output_dir(PARENT_DIR, 'annotation', change_dir=True)

DATA_DIR = PARENT_DIR / 'scviva_2'
print(f"DATA_DIR is {DATA_DIR}")


# In[ ]:


resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


# # Manual annotation of coarse cell types

# In[ ]:


adata = sc.read_h5ad(DATA_DIR / 'scviva_clustered.h5ad')
adata


# In[ ]:


# resolution 0.5
leiden_key = 'leiden0_5'
leiden_map = {
    '0' : 'Tumor',
    '1' : 'Vascular',
    '2' : 'Fibroblast',
    '3' : 'Tumor',
    '4' : 'Tumor',
    '5' : 'Tumor',
    '6' : 'Tumor',
    '7' : 'Necrosis',
    '8' : 'Myeloid',
    '9' : 'Tumor',
    '10' : 'Tumor',
    '11' : 'Tumor',
    '12' : 'Tumor',
    '13' : 'Tumor',
    '14' : 'Tumor',
}
adata.obs['scviva_coarse_ct'] = adata.obs[leiden_key].map(leiden_map)
sc.pl.umap(adata, color='scviva_coarse_ct', save='scviva_coarse_ct.png')


# In[ ]:


adata.write_h5ad('scviva_coarse_ct.h5ad')


# # Immune subclustering

# In[ ]:


output_dir = jpascvi.create_output_dir(PARENT_DIR, 'annotation', change_dir=True)
adata = sc.read_h5ad('scviva_coarse_ct.h5ad')
adata


# In[ ]:


# Clean up to avoid errors with subclustering
adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if 'leiden' in col])
adata.uns = {k: v for k, v in adata.uns.items() if 'leiden' not in k}
print("Cleaned up!")
print(adata)


# In[ ]:


# Making an output directory using the pathlib package
sub_dir = jpascvi.create_output_dir(output_dir, 'immune', change_dir=True)

immune = adata[adata.obs['scviva_coarse_ct'].isin(['Necrosis', 'Myeloid'])].copy()
print(immune)
del adata


# In[ ]:


sc.pp.neighbors(immune, use_rep="X_scVIVA")
sc.tl.umap(immune, min_dist=0.3)

for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(immune, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key, random_state=SEED)
    jpascvi.plot_umap(immune, resolution)
    jpascvi.sc_degs(immune, resolution, plots=['dotplot','matrixplot'], use_rep='X_scVIVA')

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(immune, resolutions, scores=['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_scVIVA')

immune.write_h5ad('immune_subclustered.h5ad')


# In[ ]:


qc_labels = ['component', 'volume', 'surface_area', 
             'n_genes_by_counts', 'log1p_n_genes_by_counts', 
             'total_counts', 'log1p_total_counts', 
             'n_counts', 'n_genes', 'Section', 'Patient']
sc.pl.umap(immune, color=qc_labels, frameon=False, save='qc.png')

myeloid_markers = ['RGS5', 'PDGFRB', 
                   'LUM', 'COL1A1', 
                   'CD68', 'CD163', 'CD74', 'HLA-DRA', 'CD4', 
                   'MYH11', 'ACTA2', 
                   'PDCD1', 'TLR2', 'TLR4', 'CD14', 'CD274', 
                   'FCGR3A', 'CD40', 'CD80', 'KIT',
                   'IL1B', 'ITGAX', 'VCAN', 'IFNG',
                   'WARS1', 'STAT1', 'IDO1',
                   'MS4A1', 'CD79A', 'CD19']
sc.pl.umap(immune, color=myeloid_markers, vmax='p98', use_raw=False, save='myeloid_featureplot.png')
sc.pl.umap(immune, color=myeloid_markers, vmax='p98', layer='generated_expression', save='myeloid_featureplot_corr.png')

lymphoid_markers = ['CD2', 'CD3E', 'CD4', 'CD8A', 
                    'FOXP3', 'IL7R', 'MS4A1', 'CD19', 'CD79A',
                    'KIT', 'ACTA2', 'MYH11', 'TAGLN', 
                    'VIM', 'COL1A1', 'FN1', 'IL1B', 'VEGFA']
sc.pl.umap(immune, color=lymphoid_markers, vmax='p98', use_raw=False, save='Tcells_featureplot.png')
sc.pl.umap(immune, color=lymphoid_markers, vmax='p98', layer='generated_expression', save='lymphoid_featureplot_corr.png')


# In[ ]:


del immune


# # Fibroblast subclustering

# In[ ]:


output_dir = jpascvi.create_output_dir(PARENT_DIR, 'annotation', change_dir=True)
adata = sc.read_h5ad('scviva_coarse_ct.h5ad')
adata


# In[ ]:


# Clean up to avoid errors with subclustering
adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if 'leiden' in col])
adata.uns = {k: v for k, v in adata.uns.items() if 'leiden' not in k}
print("Cleaned up!")
print(adata)


# In[ ]:


# Making an output directory using the pathlib package
sub_dir = jpascvi.create_output_dir(output_dir, 'fibroblast', change_dir=True)

fibro = adata[adata.obs['scviva_coarse_ct'] == 'Fibroblast'].copy()
print(fibro)
del adata


# In[ ]:


sc.pp.neighbors(fibro, use_rep="X_scVIVA")
sc.tl.umap(fibro, min_dist=0.3)

for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(fibro, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key, random_state=SEED)
    jpascvi.plot_umap(fibro, resolution)
    jpascvi.sc_degs(fibro, resolution, plots=['dotplot', 'matrixplot',], use_rep='X_scVIVA')

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(fibro, resolutions, scores=['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_scVIVA')

fibro.write_h5ad('fibro_subclustered.h5ad')


# In[ ]:


qc_labels = ['component', 'volume', 'surface_area', 
             'n_genes_by_counts', 'log1p_n_genes_by_counts', 
             'total_counts', 'log1p_total_counts', 
             'n_counts', 'n_genes', 'Section', 'Patient']
sc.pl.umap(fibro, color=qc_labels, frameon=False, save='qc.png')

fibro_markers = ['RGS5', 'PDGFRB', 'TAGLN', 'COL1A1', 'LUM', 'DPT', 'THY1', 'POSTN', 'MYH11', 'ACTA2']
sc.pl.umap(fibro, color=fibro_markers, vmax='p98', use_raw=False, save='fibro_featureplot.png')
sc.pl.umap(fibro, color=fibro_markers, vmax='p98', layer='generated_expression', save='fibro_featureplot_corr.png')


# # Tumor subclustering

# In[ ]:


output_dir = jpascvi.create_output_dir(PARENT_DIR, 'annotation', change_dir=True)
adata = sc.read_h5ad('scviva_coarse_ct.h5ad')
adata


# In[ ]:


# Clean up to avoid errors with subclustering
adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if 'leiden' in col])
adata.uns = {k: v for k, v in adata.uns.items() if 'leiden' not in k}
print("Cleaned up!")
print(adata)


# In[ ]:


# Making an output directory using the pathlib package
sub_dir = jpascvi.create_output_dir(output_dir, 'tumor', change_dir=True)
tumor = adata[adata.obs['scviva_coarse_ct'] == 'Tumor'].copy()
print(tumor)
del adata


# In[ ]:


sc.pp.neighbors(tumor, use_rep="X_scVIVA")
sc.tl.umap(tumor, min_dist=0.3)

for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(tumor, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key, random_state=SEED)
    jpascvi.plot_umap(tumor, resolution)
    jpascvi.sc_degs(tumor, resolution, plots=['dotplot', 'matrixplot'], use_rep='X_scVIVA')

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(tumor, resolutions, scores=['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_scVIVA')

tumor.write_h5ad('tumor_subclustered.h5ad')


# In[ ]:


qc_labels = ['component', 'volume', 'surface_area', 
             'n_genes_by_counts', 'log1p_n_genes_by_counts', 
             'total_counts', 'log1p_total_counts', 
             'n_counts', 'n_genes', 'Section', 'Patient']
sc.pl.umap(tumor, color=qc_labels, frameon=False, save='qc.png')

tumor_markers = ['CD9', 'SOX10', 'FLNB', 'DST', 
                 'ACTA2', 'DES', 'ESR1', 'PGR', 
                 'MYH11', 'TAGLN', 'MMP2', 'THBS2', 
                 'MYLK', 'SDC1', 'EGFR', 'PFN2', 
                 'PDGFRB', 'VIM', 'COL1A1', 'POSTN', 
                 'FN1', 'IL1B', 'VEGFA', 'SLC2A1']
sc.pl.umap(tumor, color=tumor_markers, vmax='p98', use_raw=False, save='tumor_featureplot.png')
sc.pl.umap(tumor, color=tumor_markers, vmax='p98', layer='generated_expression', save='tumor_featureplot_corr.png')


# In[ ]:


sc.pl.umap(tumor, color=['Section', 'Patient'], save='section_patient.png')
sc.pl.umap(tumor, color=['CD68', 'CD163'], size=0.2, use_raw=False, vmax='p98', save='cd68_and_cd163.png')
sc.pl.umap(tumor, color=['CD68', 'CD163'], size=0.2, layer='generated_expression', vmax='p98', save='cd68_and_cd163_corr.png')


# In[ ]:


# tumor = sc.read_h5ad('tumor_subclustered.h5ad')
# tumor


# In[ ]:


# sc.pl.dotplot(tumor, ['HLA-DRA', 'STAT1', 'STAT3', 'STAT4'], groupby='leiden0_7')


# In[ ]:


# # resolution 0.7
# leiden_map = {
#     '0' : 'SMC-like Tumor',
#     '1' : 'ESR1 PGR AR Tumor',
#     '2' : 'SMC-like Tumor',
#     '3' : 'SMC-like Tumor',
#     '4' : 'SMC-like Tumor',
#     '5' : 'SMC-like Tumor',
#     '6' : 'MHCII Tumor',
#     '7' : 'COL1A1 POSTN Tumor',
#     '8' : 'SMC-like Tumor', 
#     '9' : 'SDC1 EGFR Tumor',
#     '10' : 'SDC1 PDGFRB Tumor',
#     '11' : 'Ischemic Tumor',
#     '12' : 'SOX10 THBS2 Tumor',
#     '13' : 'SOX10 THBS2 Tumor',
# }
# tumor.obs['celltype'] = tumor.obs['leiden0_7'].map(leiden_map)
# sc.pl.umap(tumor, color='celltype', save='tumor_celltype.png')


# In[ ]:


# markers = ['SOX10', 'MMP2', 'THBS2', 'NCAM1', 'CHI3L1', 'ESR1', 'PGR', 'AR', 'SDC1', 'EGFR', 'PDGFRB', 'RGS5', 
#            'CCND1', 'MGP', 'LUM', 'COL1A1', 'PFN2', 'POSTN', 'VEGFA', 'SLC2A1', 'HLA-DRA', 'CD74', 'CD68', 'CD163', 
#            'MYH11', 'DES', 'ACTA2', 'MYLK', 'ACTG2', 'TAGLN', 'FLNB', 'DST', 'FN1', 'TNC', 'SCD', 'MKI67', 'TOP2A', 'RRM2', 'SYNM', 'VIM', 
#            'MAPK1', 'HELLS', 'CAV1', 'DGKG', 'IQGAP3', 'ZWINT', 'ENAH', 'CSPG4', 'FSCN1', 'GPR183', 'TOMM7', 'POLR2A', 'WARS1', 'STAT1',
#           ]
# sc.pl.dotplot(tumor, markers, 'leiden0_7', standard_scale='var', save='tumor_allmarkers_dp.png')


# In[ ]:


# markers = ['LUM', 'DPT', 'COL1A1', 'POSTN', 'PDGFRA', 'MYH11', 'MYLK', 'ACTA2']
# sc.pl.dotplot(tumor, markers, 'celltype',)


# In[ ]:


# markers = ["DGKG","FBLN1","MAPK1","PRDM1","CDH1","MLPH","MYBPC1","LAMC3","PDCD1LG2","KIT","ITGAX","CD274"]
# sc.pl.dotplot(tumor, markers, 'celltype',)


# In[ ]:


# sc.pl.dotplot(tumor, ['COL1A1', 'POSTN', 'FBLN1', 'PFN2', 'DST'], 'celltype')


# In[ ]:


# markers = ["NRXN1","INPP4B","CD36","ADGRL4","KLRD1","HAVCR2","MRC1","EPCAM","TPD52","ITGAM",
#            "IKZF2","LAMC3","IDO1","VWF","STAT4","ANKRD30A","CD247","CD4","PADI2","CSF1R","MET",
#            "PDE4A","NCR1","CXCL13","CD70","CACNA1H","CPB1","BPIFB1","PECAM1","SCUBE2",
#            "DSP","CD38","IL2RA","CLEC9A","CD80","CD86",
#            "GABRP","SELE","LPL","MUCL1","CD8A","IL2RB","MMRN1","LTF","TNFRSF9","LTBP2","CEACAM5","KDR","ESM1"]
# sc.pl.dotplot(tumor, markers, 'celltype', standard_scale='var', save='cluster7.png')


# In[ ]:


# markers = [
#     'CHI3L1', 'NCAM1',
#     'COL1A1', 'POSTN',
#     'ESR1', 'PGR', 
#     'FLNB', 'PFN2',
#     'VEGFA', 'SLC2A1',
#     'HLA-DRA', 'CD74',
#     'MYH11', 'TAGLN',
#     'SDC1', 'EGFR',
#     'SDC1', 'PDGFRB',
#     'SOX10', 'THBS2',
#     'SYNM', 'DES',
# ]
# sc.pl.dotplot(tumor, markers, 'celltype', standard_scale='var', save='old_tumor_celltype_dp.png')


# In[ ]:


# markers = [
#     'COL1A1', 'POSTN',
#     'ESR1', 'PGR', 
#     'HLA-DRA', 'CD74',
#     'SDC1', 'EGFR', 'PDGFRB',
#     'MYH11', 'TAGLN',
#     'SOX10', 'THBS2',
# ]
# sc.pl.dotplot(tumor, markers, groupby='celltype', standard_scale='var', save='tumor_celltype_dp.png')
# sc.pl.umap(tumor, color='celltype', save='tumor_celltype_umap.png')
# sc.pl.matrixplot(tumor, markers, 'celltype', standard_scale='var', save='tumor_celltype_mp.png')


# In[ ]:


# tumor.write_h5ad('tumor_annotated.h5ad')


# In[ ]:


# genes = tumor.var_names.tolist()
# genes.sort()
# print(*genes)


# In[ ]:


# markers = ["NCAM1", "NRXN1", "PFN2", "PVALB", "SNCA", "SNCG", "SNPH", "PNMT", "CX3CL1", "THY1", "CACNA1H", "NR2F1"]
# sc.pl.dotplot(tumor, markers, groupby='celltype', standard_scale='var', save='neuronal_markers.png')


# In[ ]:


# adata.obs['tumor_subtype'] = 'NA'
# tumor.obs.rename(columns={'celltype' : 'tumor_subtype'}, inplace=True)
# adata.obs.update(tumor.obs['tumor_subtype'])


# In[ ]:


# smc_pos = tumor[tumor.obs['tumor_subtype'].isin(['ESR1 PGR AR Tumor', 'MHCII Tumor', 'SMC-like Tumor'])].copy()
# smc_neg = tumor[tumor.obs['tumor_subtype'].isin(['COL1A1 POSTN Tumor', 'Ischemic Tumor', 'SDC1 EGFR Tumor', 'SDC1 PDGFRB Tumor', 'SOX10 THBS2 Tumor'])].copy()


# In[ ]:


# sc.pl.dotplot(tumor, ['FLNB', 'DST', 'SPOP', 'PDE4D', 'TNC', 'ZEB1', 'ZEB2', 'ENAH'], 'tumor_subtype')


# In[ ]:


# sc.pl.dotplot(smc_neg, ['FLNB', 'DST', 'SPOP', 'PDE4D', 'TNC', 'ZEB1', 'ZEB2', 'ENAH'], 'tumor_subtype')


# ## Endothelial and Pericyte subclustering

# In[ ]:


output_dir = jpascvi.create_output_dir(PARENT_DIR, 'annotation', change_dir=True)
adata = sc.read_h5ad('scviva_coarse_ct.h5ad')
adata


# In[ ]:


# Clean up to avoid errors with subclustering
adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if 'leiden' in col])
adata.uns = {k: v for k, v in adata.uns.items() if 'leiden' not in k}
print("Cleaned up!")
print(adata)


# In[ ]:


# Making an output directory using the pathlib package
sub_dir = jpascvi.create_output_dir(output_dir, 'vascular', change_dir=True)

vasc = adata[adata.obs['scviva_coarse_ct'] == 'Vascular'].copy()
print(vasc)
del adata


# In[ ]:


sc.pp.neighbors(vasc, use_rep="X_scVIVA")
sc.tl.umap(vasc, min_dist=0.3)

for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(vasc, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key, random_state=SEED)
    jpascvi.plot_umap(vasc, resolution)
    jpascvi.sc_degs(vasc, resolution, plots=['dotplot', 'matrixplot',], use_rep='X_scVIVA')

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(vasc, resolutions, scores=['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_scVIVA')

vasc.write_h5ad('vasc_subclustered.h5ad')


# In[ ]:


qc_labels = ['component', 'volume', 'surface_area', 
             'n_genes_by_counts', 'log1p_n_genes_by_counts', 
             'total_counts', 'log1p_total_counts', 
             'n_counts', 'n_genes', 'Section', 'Patient']
sc.pl.umap(vasc, color=qc_labels, frameon=False, save='qc.png')

vasc_markers = ['RGS5', 'PDGFRB', 'TAGLN', 'PECAM1', 'VWF', 'LYVE1', 'MYH11', 'ACTA2']
sc.pl.umap(vasc, color=vasc_markers, vmax='p98', use_raw=False, save='vasc_featureplot.png')
sc.pl.umap(vasc, color=vasc_markers, vmax='p98', layer='generated_expression', save='vasc_featureplot_corr.png')


# In[ ]:


# vasc = sc.read_h5ad('vasc_subclustered.h5ad')
# vasc


# In[ ]:


# sc.pl.dotplot(vasc, ['RGS5', 'PDGFRB', 'ACTA2', 'MYH11', 'PECAM1', 'VWF', 'CD34', 'LUM', 'COL1A1', 'DPT'], 'leiden0_4', standard_scale='var')


# In[ ]:


# # annotating cell types
# # resolution 0.4

# leiden_map = {
#     '0' : 'Pericyte',
#     '1' : 'Pericyte', 
#     '2' : 'Pericyte',
#     '3' : 'Pericyte',
#     '4' : 'Endothelial',
# }
# vasc.obs['celltype'] = vasc.obs['leiden0_4'].map(leiden_map)
# sc.pl.umap(vasc, color='celltype', save='celltype.png')


# In[ ]:


# adata.obs.update(vasc.obs['celltype'])
# adata.obs['celltype']


# # Immune annotation

# In[ ]:


# sub_dir = jpascvi.create_output_dir(output_dir, 'immune', change_dir=True)
# immune = sc.read_h5ad('immune_subclustered.h5ad')
# immune


# In[ ]:


# markers = ['RGS5', 'PDGFRB', 
#            'LUM', 'COL1A1', 
#            'CD68', 'CD163', 'CD74', 'HLA-DRA', 'CD4', 
#            'CD2', 'CD3D', 'CD3E', 'CD8A', 
#            'MYH11', 'ACTA2', 'MYLK', 'TAGLN',
#            'PDCD1', 'TLR2', 'TLR4', 'CD14', 'CD274', 
#            'FCGR3A', 'CD40', 'CD80', 'KIT',
#            'IL1B', 'ITGAX', 'VCAN', 'IFNG',
#            'WARS1', 'STAT1', 'IDO1',
#            'MS4A1', 'CD79A', 'CD19',
#            'ADIPOQ', 'PLIN1', 'LPL',
#           ]
# sc.pl.dotplot(immune, markers, 'leiden1_0', standard_scale='var')


# In[ ]:


# # resolution 1.0
# leiden_key = 'leiden1_0'
# leiden_map = {
#     '0' : 'B',
#     '1' : 'T_and_NK',
#     '2' : 'Pericyte',
#     '3' : 'T_and_NK',
#     '4' : 'Tumor',
#     '5' : 'Necrosis',
#     '6' : 'Macrophage',
#     '7' : 'Tumor',
#     '8' : 'Tumor',
#     '9' : 'Macrophage',
#     '10' : 'Tumor',
#     '11' : 'Tumor',
#     '12' : 'Macrophage',
#     '13' : 'Tumor',
#     '14' : 'Tumor',
#     '15' : 'Mast',
# }
# immune.obs['celltype'] = immune.obs[leiden_key].map(leiden_map)
# sc.pl.umap(immune, color='celltype', save='immune_celltype.png')


# In[ ]:


# adata.obs['celltype'] = 'Unknown'
# adata.obs.update(immune.obs['celltype'])
# adata.obs['celltype']


# In[ ]:


# immune.obs['ann_leiden'] = 'immune' + immune.obs[leiden_key].astype(str)
# adata.obs['ann_leiden'] = 'Unknown'
# adata.obs.update(immune.obs['ann_leiden'])
# adata.obs['ann_leiden']


# In[ ]:


# del immune


# # Fibroblast annotation

# In[ ]:


# sub_dir = jpascvi.create_output_dir(output_dir, 'fibroblast', change_dir=True)
# fibro = sc.read_h5ad('fibro_subclustered.h5ad')
# fibro


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Back to the combined object
# - Note - will need to add back the other clusters that were discovered after subclustering

# In[ ]:


# os.chdir(output_dir)
# sc.settings.figdir = output_dir

# sc.pl.umap(adata, color='celltype', save='celltype.png')


# In[ ]:


# adata.write_h5ad('annotated.h5ad')


# In[ ]:


# print(adata.obs.celltype.cat.categories.tolist())


# In[ ]:


# markers = [
#     'ADIPOQ', 'PLIN1',
#     'VWF', 'PECAM1', 
#     'LUM', 'COL1A1',
#     'CD68', 'CD163',  
#     'KIT',
#     'IL1B', 
#     'RGS5', 'PDGFRB',
#     'CD2', 'CD3D', 
#     'MYH11', 'ACTA2', 'DES', 'TAGLN',
# ]

# sc.pl.dotplot(adata, var_names=markers, groupby='celltype', dendrogram=False, save='celltype_dotplot.png', standard_scale='var')


# # Making figures

# In[ ]:


# data_dir = parent_dir / 'objects' # use the properly colored objects that were made in the bysection_v3 notebook
# adata = sc.read_h5ad(data_dir / 'all_cells.h5ad')
# adata


# In[ ]:


# umap = sc.pl.umap(adata, color='celltype', frameon=False, show=False, return_fig=True, title='', size=0.2)
# umap.set_size_inches(7, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig('allcells.png', dpi=300, bbox_inches='tight')
# plt.close()

# umap = sc.pl.umap(adata, color='celltype', frameon=False, show=False, return_fig=True, title='', size=0.2)
# umap.set_size_inches(7, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig('allcells.pdf', dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:


# markers = [
#     'ADIPOQ', 'PLIN1',
#     'VWF', 'PECAM1', 
#     'LUM', 'COL1A1',
#     'CD68', 'CD163',  
#     'KIT',
#     'IL1B', 
#     'RGS5', 'PDGFRB',
#     'CD2', 'CD3D', 
#     'MYH11', 'ACTA2', 'DES', 'TAGLN',
# ]
# sc.pl.dotplot(adata, var_names=markers, groupby='celltype', dendrogram=False, standard_scale='var', save='allcells_dp.png')
# sc.pl.dotplot(adata, var_names=markers, groupby='celltype', dendrogram=False, standard_scale='var', save='allcells_dp.pdf')


# In[ ]:


# sub_dir = jpascvi.create_output_dir(output_dir, 'tumor', change_dir=True)


# In[ ]:


# tumor = sc.read_h5ad(data_dir / 'tumor.h5ad')
# tumor


# In[ ]:


# umap = sc.pl.umap(tumor, color='tumor_subtype', frameon=False, show=False, return_fig=True, title='', size=0.2)
# umap.set_size_inches(7, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig('tumor.png', dpi=300, bbox_inches='tight')
# plt.close()

# umap = sc.pl.umap(tumor, color='tumor_subtype', frameon=False, show=False, return_fig=True, title='', size=0.2)
# umap.set_size_inches(7, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig('tumor.pdf', dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:


# markers = [
#     'ESR1', 'PGR',
#     'MYH11', 'TAGLN',
#     'SDC1', 'EGFR',
#     'HLA-DRA', 'CD74',
#     'VEGFA', 'SLC2A1',
#     'COL1A1', 'POSTN',
#     'SDC1', 'PDGFRB',
#     'SOX10', 'THBS2',
# ]
# sc.pl.dotplot(tumor, var_names=markers, groupby='tumor_subtype', dendrogram=False, standard_scale='var', save='tumor_dp.png')
# sc.pl.dotplot(tumor, var_names=markers, groupby='tumor_subtype', dendrogram=False, standard_scale='var', save='tumor_dp.pdf')


# In[ ]:


# umap = sc.pl.umap(tumor, color='tumor_subtype', groups='ESR1 PGR AR Tumor', frameon=False, show=False, return_fig=True, title='', size=0.5)
# umap.set_size_inches(7, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig('esr_pgr_highlight.png', dpi=300, bbox_inches='tight')
# plt.close()

# umap = sc.pl.umap(tumor, color='tumor_subtype', groups='ESR1 PGR AR Tumor', frameon=False, show=False, return_fig=True, title='', size=0.5)
# umap.set_size_inches(7, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig('esr_pgr_highlight.pdf', dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:


# # hide the legend

# umap = sc.pl.umap(tumor, color='tumor_subtype', groups='ESR1 PGR AR Tumor', frameon=False, show=False, return_fig=True, title='', size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend('', frameon=False) # Remove the legend
# plt.tight_layout()
# plt.savefig('esr_pgr_highlight_nolegend.png', dpi=300, bbox_inches='tight')
# plt.close()

# umap = sc.pl.umap(tumor, color='tumor_subtype', groups='ESR1 PGR AR Tumor', frameon=False, show=False, return_fig=True, title='', size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend('', frameon=False) # Remove the legend
# plt.tight_layout()
# plt.savefig('esr_pgr_highlight_nolegend.pdf', dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:


# gene = 'ESR1'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()

# gene = 'PGR'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()

# gene = 'AR'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:


# gene = 'MYH11'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()

# gene = 'TAGLN'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()

# gene = 'CD44'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()

# gene = 'RSPO3'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()

# gene = 'STAT3'
# umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
# umap.set_size_inches(5, 4)
# plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.tight_layout()
# plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
# plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:




