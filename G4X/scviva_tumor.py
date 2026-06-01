#!/usr/bin/env python
# coding: utf-8

# # Tumor subset only - clustering and manually annotating the scVIVA model of the ULMS G4X dataset
# - annotates the tumor subset of the scVIVA_2 clustered object (proseg). The one with the reduced learning rate.
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
output_dir = jpascvi.create_output_dir(PARENT_DIR, 'scviva_tumor', change_dir=True)

data_dir = PARENT_DIR / 'annotation'
print(data_dir)


# # Clustering

# In[ ]:


jpa_markers = jpascvi.import_markers((PARENT_DIR / 'ref/jpa_g4x_breast_panel.csv'), output_type='dict')
jpa_markers = {key: value for key, value in jpa_markers.items() if key != 'Plasma_cell'} # JCHAIN and IGHG1 not in this segmentation run
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
               1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


# In[ ]:


adata = sc.read_h5ad(data_dir / 'scviva_celltype.h5ad')
adata


# In[ ]:


cols_to_drop = ['_indices', '_scvi_batch', '_scvi_ind_x', '_scvi_labels', 
                'leiden0_1', 'leiden0_2', 'leiden0_3', 'leiden0_4', 'leiden0_5', 
                'leiden0_6', 'leiden0_7', 'leiden0_8', 'leiden0_9', 'leiden1_0', 
                'leiden1_1', 'leiden1_2', 'leiden1_3', 'leiden1_4', 'leiden1_5']
adata.obs.drop(cols_to_drop, axis='columns', inplace=True, errors='ignore')

uns_to_drop = ['_scvi_manager_uuid', '_scvi_uuid', 
               'dendrogram_leiden0_1', 'dendrogram_leiden0_2', 'dendrogram_leiden0_3', 'dendrogram_leiden0_4', 
               'dendrogram_leiden0_5', 'dendrogram_leiden0_6', 'dendrogram_leiden0_7', 'dendrogram_leiden0_8', 
               'dendrogram_leiden0_9', 'dendrogram_leiden1_0', 'dendrogram_leiden1_1', 'dendrogram_leiden1_2', 
               'dendrogram_leiden1_3', 'dendrogram_leiden1_4', 'dendrogram_leiden1_5', 
               'leiden0_1', 'leiden0_1_colors', 'leiden0_2', 'leiden0_2_colors', 'leiden0_3', 'leiden0_3_colors', 
               'leiden0_4', 'leiden0_4_colors', 'leiden0_5', 'leiden0_5_colors', 'leiden0_6', 'leiden0_6_colors', 
               'leiden0_7', 'leiden0_7_colors', 'leiden0_8', 'leiden0_8_colors', 'leiden0_9', 'leiden0_9_colors', 
               'leiden1_0', 'leiden1_0_colors', 'leiden1_1', 'leiden1_1_colors', 'leiden1_2', 'leiden1_2_colors', 
               'leiden1_3', 'leiden1_3_colors', 'leiden1_4', 'leiden1_4_colors', 'leiden1_5', 'leiden1_5_colors', 
               'log1p', 'neighbors', 'rank_genes_groups', 'umap']
for key in uns_to_drop:
    if key in adata.uns:
        adata.uns.pop(key, None)
    
del adata.obsp['connectivities']
del adata.obsp['distances']
del adata.obsm['X_umap']

print(adata)


# In[ ]:


# subset for tumor cells
adata = adata[adata.obs['celltype'] == 'Tumor'].copy()
print(np.unique(adata.obs['celltype']))
print(adata)


# In[ ]:


print('Calculating neighbors')
sc.pp.neighbors(adata, use_rep="X_scVIVA", n_neighbors=30)
print('Calculating UMAP')
sc.tl.umap(adata, min_dist=0.3)


# In[ ]:


sc.pl.umap(adata, color='coarse_celltype', frameon=False, save='coarse_celltype.png')
sc.pl.umap(adata, color=['Section', 'Patient'], frameon=False, save='section_and_patient.png')
sc.pl.umap(adata, color=['component', 'volume', 'surface_area', 
                         'n_genes_by_counts', 'log1p_n_genes_by_counts', 
                         'total_counts', 'log1p_total_counts', 'n_counts', 'n_genes',], 
           frameon=False, save='qc.png')


# In[ ]:


jpascvi.featureplot(adata, jpa_markers, save='scviva.png', use_raw=False, vmax='p98') # False to allow scaled values to be plotted from adata.X


# In[ ]:


# Leiden clustering at multiple resolutions and differential gene expression
for resolution in resolutions:
    str_res = str(resolution).replace('.', '_')
    leiden_key = 'leiden' + str_res
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
    jpascvi.plot_umap(adata, resolution)
    jpascvi.sc_degs(adata, resolution, use_rep='X_scVIVA', plots=['dotplot'])

# Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
jpascvi.cluster_stats(adata, resolutions, scores = ['Calinski-Harabasz', 'Davies-Bouldin'], rep='X_scVIVA')


# In[ ]:


adata.write_h5ad('scviva_tumor_clustered.h5ad')


# In[ ]:


sc.pl.umap(adata, 
           color=['MYH11', 'MYLK', 'DES', 'TAGLN', 'ESR1', 'COL1A1', 'SDC1', 'SOX10'], 
           size=0.2, 
           save='ulms_feature_plot.png', 
           vmax='p98')

sc.pl.umap(adata, 
           color=['VEGFA', 'SLC2A1',], 
           size=0.2, 
           save='ischemia_feature_plot.png', 
           vmax='p98')

sc.pl.umap(adata, 
           color=['VWF', 'PECAM1', 'COL1A1', 'LUM', 'CD68', 'CD163', 'RGS5', 'PDGFRB', 'CD2', 'IL7R'], 
           size=0.2, 
           save='nontumor_feature_plot.png', 
           vmax='p98')

sc.pl.umap(adata, 
           color=['ESR1', 'PGR', 'AR'], 
           save='hormononal_feature_plot.png', 
           size=0.2, 
           vmax='p98')

sc.pl.umap(adata, 
           color=['LAG3', 'PDCD1'], 
           save='immune_checkpoint_feature_plot.png', 
           size=0.2, 
           vmax='p98')


# In[ ]:


sc.pl.umap(adata, 
           color=['MYH11', 'MYLK', 'DES', 'TAGLN', 'ESR1', 'COL1A1', 'SDC1', 'SOX10'], 
           size=0.2, 
           save='ulms_feature_plot_corr.png', 
           vmax='p98',
           layer='generated_expression'
          )

sc.pl.umap(adata, 
           color=['VEGFA', 'SLC2A1',], 
           size=0.2, 
           save='ischemia_feature_plot_corr.png', 
           vmax='p98',
           layer='generated_expression'
          )

sc.pl.umap(adata, 
           color=['VWF', 'PECAM1', 'COL1A1', 'LUM', 'CD68', 'CD163', 'RGS5', 'PDGFRB', 'CD2', 'IL7R'], 
           size=0.2, 
           save='nontumor_feature_plot_corr.png', 
           vmax='p98',
           layer='generated_expression'
          )

sc.pl.umap(adata, 
           color=['ESR1', 'PGR', 'AR'], 
           save='hormononal_feature_plot_corr.png', 
           size=0.2, 
           vmax='p98',
           layer='generated_expression'
          )

sc.pl.umap(adata, 
           color=['LAG3', 'PDCD1'], 
           save='immune_checkpoint_feature_plot_corr.png', 
           size=0.2, 
           vmax='p98',
           layer='generated_expression'
          )


# In[ ]:


sc.pl.umap(adata, 
           color=['MYH11', 'MYLK', 'DES', 'TAGLN', 'ESR1', 'COL1A1', 'SDC1', 'SOX10'], 
           size=0.2, 
           save='ulms_feature_plot_norm.png', 
           vmax='p98',
           layer='X_normalized_resolVI'
          )

sc.pl.umap(adata, 
           color=['VEGFA', 'SLC2A1',], 
           size=0.2, 
           save='ischemia_feature_plot_norm.png', 
           vmax='p98',
           layer='X_normalized_resolVI'
          )

sc.pl.umap(adata, 
           color=['VWF', 'PECAM1', 'COL1A1', 'LUM', 'CD68', 'CD163', 'RGS5', 'PDGFRB', 'CD2', 'IL7R'], 
           size=0.2, 
           save='nontumor_feature_plot_norm.png', 
           vmax='p98',
           layer='X_normalized_resolVI'
          )

sc.pl.umap(adata, 
           color=['ESR1', 'PGR', 'AR'], 
           save='hormononal_feature_plot_norm.png', 
           size=0.2, 
           vmax='p98',
           layer='X_normalized_resolVI'
          )

sc.pl.umap(adata, 
           color=['LAG3', 'PDCD1'], 
           save='immune_checkpoint_feature_plot_norm.png', 
           size=0.2, 
           vmax='p98',
           layer='X_normalized_resolVI'
          )


# In[ ]:


gene = 'ESR1'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()

gene = 'PGR'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()

gene = 'AR'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()


# In[ ]:


gene = 'MYH11'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()

gene = 'TAGLN'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()

gene = 'CD44'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()

gene = 'RSPO3'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()

gene = 'STAT3'
umap = sc.pl.umap(tumor, color=gene, color_map='inferno', frameon=False, show=False, return_fig=True, title=gene, size=0.5)
umap.set_size_inches(5, 4)
plt.legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{gene}.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{gene}.png', dpi=300, bbox_inches='tight')
plt.close()


# # Manual annotation

# In[ ]:


# tumor = sc.read_h5ad(output_dir / 'scviva_tumor_clustered.h5ad')
# tumor


# In[ ]:


# leiden_key = 'leiden1_5'
# sc.pl.umap(tumor, color=leiden_key, groups=['1', '3', '14', '16'], palette=['red'])


# In[ ]:


# leiden_key = 'leiden1_5'
# markers = [
#     'ESR1', 'PGR', 'AR', 'RSPO3', 'CCND1', 'CD44',
#     'MYH11', 'TAGLN', 'ACTA2', 'ACTG2', 'MYLK', 'A2M',
#     'TNC', 'DST', 'FLNB', 'SYNM',
#     'CHI3L1', 'NCAM1',
#     'THBS2', 'PFN2',
#     'HLA-DRA', 'CD74',
#     'CD68', 'CD163',
#     'WARS1', 'STAT1',
#     'VEGFA', 'SLC2A1',
#     'COL1A1', 'POSTN',
#     'SDC1', 'PDGFRB', 'EGFR', 'CD9',
#     'MKI67', 'TOP2A',
# ]
# sc.pl.dotplot(tumor, 
#               var_names=markers, 
#               groupby=leiden_key, 
#               dendrogram=False, 
#               standard_scale='var', 
#               save='full_ann_dotplot.png')
# sc.pl.dotplot(tumor, 
#               var_names=markers, 
#               groupby=leiden_key, 
#               dendrogram=False, 
#               standard_scale='var', 
#               use_raw=False, 
#               save='full_ann_dotplot_scaled.png')
# sc.pl.dotplot(tumor, 
#               var_names=markers, 
#               groupby=leiden_key, 
#               dendrogram=False, 
#               standard_scale='var', 
#               layer='X_normalized_resolVI', 
#               save='full_ann_dotplot_norm.png')
# sc.pl.dotplot(tumor, 
#               var_names=markers, 
#               groupby=leiden_key, 
#               dendrogram=False, 
#               standard_scale='var', 
#               layer='generated_expression', 
#               save='full_ann_dotplot_corr.png')


# In[ ]:


# # annotate
# leiden_key = 'leiden1_5'
# leiden_map = {
#     '0' : 'CHI3L1-low SMC-like Tumor',
#     '1' : 'ESR1 PGR AR Tumor',
#     '2' : 'MHCII Tumor',
#     '3' : 'ESR1 PGR AR Tumor',
#     '4' : 'CHI3L1-low SMC-like Tumor',
#     '5' : 'Cycling SMC-high Tumor',
#     '6' : 'CHI3L1-low SMC-like Tumor',
#     '7' : 'Cycling SMC-high Tumor',
#     '8' : 'MHCII Tumor',
#     '9' : 'COL1A1 POSTN Tumor',
#     '10' : 'CHI3L1-low SMC-like Tumor',
#     '11' : 'CHI3L1-high SMC-like Tumor',
#     '12' : 'MHCII Tumor',
#     '13' : 'Ischemic Tumor',
#     '14' : 'ESR1 PGR AR Tumor',
#     '15' : 'CHI3L1-low SMC-like Tumor',
#     '16' : 'ESR1 PGR AR Tumor',
#     '17' : 'Ischemic Tumor',
#     '18' : 'Ischemic Tumor',
#     '19' : 'CHI3L1-high SMC-like Tumor',
#     '20' : 'SDC1 PDGFRB Tumor',
#     '21' : 'Cycling SMC-high Tumor',
#     '22' : 'SDC1 PDGFRB Tumor',
#     '23' : 'COL1A1 POSTN Tumor',
#     '24' : 'CHI3L1-low SMC-like Tumor',
#     '25' : 'Cycling SMC-low Tumor',
# }
# tumor.obs['tumor_subtype'] = tumor.obs[leiden_key].map(leiden_map)
# tumor.obs['tumor_subtype']


# In[ ]:


# subtype_cats = [
#     "ESR1 PGR AR Tumor",
#     "CHI3L1-high SMC-like Tumor",
#     "CHI3L1-low SMC-like Tumor",
#     "MHCII Tumor",
#     "Cycling SMC-high Tumor",
#     "Cycling SMC-low Tumor",
#     "Ischemic Tumor",
#     "COL1A1 POSTN Tumor",
#     "SDC1 PDGFRB Tumor",
# ]
# tumor.obs['tumor_subtype'] = pd.Categorical(tumor.obs['tumor_subtype'], ordered=True, categories=subtype_cats)
# tumor.obs['tumor_subtype']


# In[ ]:


# subtype_colors = [
#     "#008080", # "ESR1 PGR AR"
#     "#BE469A", # "CHI3L1-high SMC-like Tumor"
#     "#e892ce", # "CHI3L1-low SMC-like Tumor"
#     "#D52927", # "MHCII Tumor"
#     "#8E5C52", # "Cycling SMC-high Tumor"
#     "#AD8981", # "Cycling SMC-low Tumor"
#     "#9168AB", # "Ischemic Tumor"
#     "#ed7015", # "COL1A1 POSTN Tumor"
#     "#fcae1f", # "SDC1 PDGFRB Tumor"
# ]
# tumor.uns['tumor_subtype_colors'] = subtype_colors


# In[ ]:


# sc.pl.umap(tumor, color='tumor_subtype', save='scviva_tumor_subtype.png', frameon=False)
# sc.pl.umap(tumor, color='tumor_subtype', save='scviva_tumor_subtype.pdf', frameon=False)


# In[ ]:


# markers = [
#     'ESR1', 'PGR',
#     'CHI3L1', 'NCAM1',
#     'MYH11', 'TAGLN',
#     'HLA-DRA', 'CD74',
#     'MKI67', 'TOP2A',
#     'VEGFA', 'SLC2A1',
#     'COL1A1', 'POSTN',
#     'SDC1', 'PDGFRB',
# ]
# sc.pl.dotplot(tumor, var_names=markers, groupby='tumor_subtype', dendrogram=False, standard_scale='var', save='ann_dp.png')
# sc.pl.dotplot(tumor, var_names=markers, groupby='tumor_subtype', dendrogram=False, standard_scale='var', save='ann_dp.pdf')


# In[ ]:


# tumor.write_h5ad(output_dir / 'tumor_annotated.h5ad')


# # Bar plots

# In[ ]:


# # order sections properly. This is how they will print on the page.
# sections = [
#     'B01', 'C01', 'D01', 'E01', 
#     'F01', 'G01', 'H01', 'A02', 
#     'B02', 'C02', 'E02', 'F02', 
#     'G02', 'H02', 'A03', 'B03', 
#     'C03', 'D03', 'E03', 'F03', 
#     'G03', 'H03', 'A04', 'B04', 
#     'C04', 'D04',
# ]


# In[ ]:


# # Making section barplot of tumor subtypes

# # Create a color mapping
# tumor_subtype_colors = tumor.uns['tumor_subtype_colors']
# color_mapping = {tumor_subtype: tumor_subtype_colors[i] 
#                  for i, tumor_subtype in enumerate(tumor_subtype_order)}
# print("Color Mapping from adata.uns:", color_mapping)

# # Get the counts of each tumor subtype in each section
# grouped = tumor.obs.groupby('Section', observed=True)['tumor_subtype'].value_counts()

# # Create a data frame from the grouped series
# df = grouped.reset_index(name='count')

# # Get the totals per section
# total = df.groupby('Section', observed=True)['count'].sum().reset_index(name='total')
# df = pd.merge(df, total, on='Section')

# # Calculate the percentages
# df['percentage'] = (df['count'] / df['total']) * 100

# # Pivot to wide format for stacked bar plot
# df_pivot = df.pivot(index='Section', columns='tumor_subtype', values='percentage')

# # Reorder columns to match desired tumor subtype order
# df_pivot = df_pivot[tumor_subtype_order] # column order
# df_pivot = df_pivot.reindex(sections[::-1])  # row order (custom section order)
# df_pivot = df_pivot.fillna(0)  # handle missing combinations

# # Ensure the colors are in the correct order
# ordered_colors = [color_mapping[t] for t in df_pivot.columns]
# print("Ordered Colors for Bar Plot:", ordered_colors)

# # Make plot
# fig, ax = plt.subplots(figsize=(8, 8))
# df_pivot.plot(
#     kind='barh',
#     stacked=True,
#     width=0.9,
#     color=ordered_colors,
#     ax=ax
# )
# ax.set_title('Tumor Subtypes in Each Section')
# ax.set_xlabel('Percentage (%)')
# ax.grid(False)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Tumor Subtype')
# # Or remove legend:
# # ax.get_legend().remove()

# plt.tight_layout()
# png_path = output_dir / 'subtypes_only_bysection_barplot.png'
# plt.savefig(png_path, dpi=300, bbox_inches='tight')
# pdf_path = output_dir / 'subtypes_only_bysection_barplot.pdf'
# plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
# plt.close()


# In[ ]:


# tumor.obs['Patient']


# In[ ]:


# tumor.obs['Patient'].cat.categories


# In[ ]:


# # G4X patient to scRNAseq patient mapping
# patient_map = {
#     '1'	: '01',
#     '2'	: '02',
#     '3'	: '03',
#     '4'	: '04',
#     '5'	: '06',
#     '6'	: '07',
#     '7'	: '08',
#     '8'	: '09',
#     '9'	: '10',
# }
# tumor.obs['Sample'] = tumor.obs['Patient'].map(patient_map)


# In[ ]:


# # Making sample barplot of tumor subtypes

# # Create a color mapping
# tumor_subtype_colors = tumor.uns['tumor_subtype_colors']
# color_mapping = {tumor_subtype: tumor_subtype_colors[i]
#                  for i, tumor_subtype in enumerate(tumor_subtype_order)}
# print("Color Mapping from adata.uns:", color_mapping)

# # Get the counts of each tumor subtype in each sample
# grouped = tumor.obs.groupby('Sample', observed=True)['tumor_subtype'].value_counts()

# # Create a data frame from the grouped series
# df = grouped.reset_index(name='count')

# # Get the totals per sample
# total = df.groupby('Sample', observed=True)['count'].sum().reset_index(name='total')
# df = pd.merge(df, total, on='Sample')

# # Calculate the percentages
# df['percentage'] = (df['count'] / df['total']) * 100

# # Pivot to wide format for stacked bar plot
# df_pivot = df.pivot(index='Sample', columns='tumor_subtype', values='percentage')

# # Reorder columns to match desired tumor subtype order
# df_pivot = df_pivot[tumor_subtype_order] # column order
# df_pivot = df_pivot.sort_index(ascending=False)
# df_pivot = df_pivot.fillna(0)  # handle missing combinations

# # Ensure the colors are in the correct order
# ordered_colors = [color_mapping[t] for t in df_pivot.columns]
# print("Ordered Colors for Bar Plot:", ordered_colors)

# # Make plot
# fig, ax = plt.subplots(figsize=(8, 8))
# df_pivot.plot(
#     kind='barh',
#     stacked=True,
#     width=0.9,
#     color=ordered_colors,
#     ax=ax
# )
# ax.set_title('Tumor Subtypes in Each Sample')
# ax.set_xlabel('Percentage (%)')
# ax.grid(False)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Tumor Subtype')
# # Or remove legend:
# # ax.get_legend().remove()

# plt.tight_layout()
# png_path = output_dir / 'subtypes_only_bysample_barplot.png'
# plt.savefig(png_path, dpi=300, bbox_inches='tight')
# pdf_path = output_dir / 'subtypes_only_bysample_barplot.pdf'
# plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
# plt.close()


# # Map back to resolvi umap

# In[ ]:


# tumor = sc.read_h5ad(output_dir / 'tumor_annotated.h5ad')
# tumor


# In[ ]:


# sc.pl.umap(tumor, color='tumor_subtype', save='scviva_tumor_subtype.png', frameon=False)
# sc.pl.umap(tumor, color='tumor_subtype', save='scviva_tumor_subtype.pdf', frameon=False)


# In[ ]:


# RESOLVI_DIR = PARENT_DIR / 'resolvi_tumor'
# resolvi_adata = sc.read_h5ad(RESOLVI_DIR / 'resolvi_tumor.h5ad')
# print(resolvi_adata)


# In[ ]:


# resolvi_adata.obs['tumor_subtype'] = tumor.obs.loc[resolvi_adata.obs.index, 'tumor_subtype']
# resolvi_adata


# In[ ]:


# sc.pl.umap(resolvi_adata, color='tumor_subtype', save='resolvi_tumor_subtype.png', frameon=False)
# sc.pl.umap(resolvi_adata, color='tumor_subtype', save='resolvi_tumor_subtype.pdf', frameon=False)


# # Extra

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




