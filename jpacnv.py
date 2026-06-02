#!/usr/bin/env python
# coding: utf-8

# Wrapper functions for infercnvpy

# In[ ]:


import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
from pathlib import Path
import matplotlib as mpl
import infercnvpy as cnv


# In[ ]:


# Function reads in all the h5ad files from directory to separate anndata objects
def import_data(directory): 
    # List all files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('.h5ad')]
    data_list = []

    for file in files:
        # Load the AnnData object
        path = os.path.join(directory, file)
        adata = sc.read_h5ad(path)
        # Append the annotated AnnData to the list
        data_list.append(adata)

    return data_list


# In[ ]:


def create_output_dir(master_dir, sub_dir_name, change_dir=False, change_figdir=False):
    '''
    Create an output directory as a subdirectory 'sub_dir_name' string within a parent directory master_dir
    Will not overwrite the files within that directory
    2025-05-28 moved to functions file
    2025-08-29 added change_dir
    2026-05-06 added change_figdir
    '''
    output_dir = master_dir / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created output directory {output_dir}')
    if change_dir:
        os.chdir(output_dir)
        sc.settings.figdir = output_dir
        print(f'Working directory and scanpy figure output directory changed to {output_dir}')
    if change_figdir:
        sc.settings.figdir = output_dir
        print(f'Default scanpy figure output directory changed to {output_dir}')
    return output_dir


# In[ ]:


# def gof_format(adata):
#     # read in gene order file created from GRCh38-2024-A gtf from 10X Genomics
#     # used the gtf_to_position_file.py script from the infercnv authors to do this
#     gof = pd.read_csv("/oak/stanford/groups/longaker/ULMS/ref/refdata-gex-GRCh38-2024-A/genes/GRCh38-2024-A.txt",
#                       sep="\t", index_col=False, header=None, names=["ensg", "chromosome", "start", "end"])
    
#     # Now we need to do some data wrangling to get the input file in the proper format for infercnv
#     # need to add back the ensg ids to adata.var and then add the gof
#     # We'll read in a sample adata with the gene_name to ensg id mapping
#     sample_adata = sc.read_h5ad("/oak/stanford/groups/longaker/ULMS/redo_analysis/preprocessed/Batch01_preproc_singlet_adata.h5ad")
    
#     ensg_map = sample_adata.var['gene_ids'].to_dict()
#     adata.var['ensg'] = adata.var.index.map(ensg_map)
#     del sample_adata
    
#     adata.var = adata.var.reset_index()
#     adata.var = adata.var.merge(gof, how="left", on="ensg")
#     adata.var = adata.var.set_index('index')
#     print(adata.var)
#     return adata


# In[ ]:


# def gof_format(adata, ref, ensg_map):
#     '''
#     2026-03-30 new version removes hard coded paths and adds ref and ensg_map arguments
#     2026-05-07 added checks
#     ref: path to gene order file created by running the gtf_to_position_file.py script from infercnv authors on GRCh38-2024-A gtf from 10X Genomics
#     ensg_map: dictionary mapping adata.var index to ENSG id
#     '''
#     # read in gene order file created from GRCh38-2024-A gtf from 10X Genomics
#     # used the gtf_to_position_file.py script from the infercnv authors to do this
#     gof = pd.read_csv(ref, sep="\t", index_col=False, header=None, names=["ensg", "chromosome", "start", "end"])
    
#     # Now we need to do some data wrangling to get the input file in the proper format for infercnv
#     # need to add back the ensg ids to adata.var and then add the gof
#     # We'll read in a sample adata with the gene_name to ensg id mapping
#     if not isinstance(ensg_map, dict):
#         print("ensg_map must be a dictionary")
#         return

#     adata.var['ensg'] = adata.var.index.map(ensg_map)
#     adata.var = adata.var.reset_index()
#     adata.var = adata.var.merge(gof, how="left", on="ensg")
#     adata.var = adata.var.set_index('index')
#     print(adata.var)
#     return adata

def gof_format(adata, ref, ensg_map):
    '''
    2026-03-30 new version removes hard coded paths and adds ref and ensg_map arguments
    ref: path to gene order file created by running the gtf_to_position_file.py script from infercnv authors on GRCh38-2024-A gtf from 10X Genomics
    ensg_map: dictionary mapping adata.var index to ENSG id
    '''
    # read in gene order file created from GRCh38-2024-A gtf from 10X Genomics
    # used the gtf_to_position_file.py script from the infercnv authors to do this
    gof = pd.read_csv(ref, sep="\t", index_col=False, header=None, names=["ensg", "chromosome", "start", "end"])
    
    if not isinstance(ensg_map, dict):
        print("ensg_map must be a dictionary")
        return

    # Check 1: Which genes in adata.var have no mapping in ensg_map
    genes_in_adata = set(adata.var_names)
    genes_in_ensg_map = set(ensg_map.keys())
    unmapped_genes = genes_in_adata - genes_in_ensg_map
    if unmapped_genes:
        print(f"Warning: {len(unmapped_genes)} genes in adata.var not found in ensg_map. These will be removed.")
        print(f"  Examples: {list(unmapped_genes)[:10]}")
    
    # Check 2: Which mapped ENSG IDs are not in the gene order file
    ensg_ids_in_gof = set(gof['ensg'])
    mapped_genes = genes_in_adata & genes_in_ensg_map
    genes_not_in_gof = {g for g in mapped_genes if ensg_map[g] not in ensg_ids_in_gof}
    if genes_not_in_gof:
        print(f"Warning: {len(genes_not_in_gof)} genes have ENSG IDs not found in gene order file. These will be removed.")
        print(f"  Examples: {list(genes_not_in_gof)[:10]}")
    
    # Subset adata to only genes that are in ensg_map AND whose ENSG IDs are in the gof
    genes_to_keep = sorted(mapped_genes - genes_not_in_gof)
    print(f"Keeping {len(genes_to_keep)} out of {len(genes_in_adata)} genes.")
    adata = adata[:, genes_to_keep].copy()
    
    # Now proceed with the merge
    adata.var['ensg'] = adata.var.index.map(ensg_map)
    adata.var = adata.var.reset_index()
    adata.var = adata.var.merge(gof, how="left", on="ensg")
    adata.var = adata.var.set_index('index')
    print(adata.var)
    return adata


# In[ ]:


def cnv_plot(adata, celltype_key='celltype', resolution=None, cnv_resolution=None):
    '''
    Wrapper for infercnvpy
    Makes chromosome heatmaps and computes CNV clusters and CNV umap
    Make sure sc.tl.umap has already been computed
    2026-04-03 added celltype_key argument
    2026-05-07 made resolution optional and added cnv_resolution
    '''
    # plots
    cnv.pl.chromosome_heatmap(adata, groupby=celltype_key, save='chr_heatmap.png')
    cnv.pl.chromosome_heatmap_summary(adata, groupby=celltype_key, save='chr_heatmap_summary.png')

    if resolution:
        str_res = 'leiden' + str(resolution).replace('.', '_')
        cnv.pl.chromosome_heatmap(adata, groupby=str_res, save='chr_heatmap_SCVIclusters.png')
        cnv.pl.chromosome_heatmap_summary(adata, groupby=str_res, save='chr_heatmap_SCVIclusters_summary.png')

    print("Computing PCA, neighbors, and leiden clusters")
    cnv.tl.pca(adata)
    cnv.pp.neighbors(adata)
    cnv.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=cnv_resolution)

    cnv.pl.chromosome_heatmap(adata, groupby="cnv_leiden", dendrogram=True, save='cnv_cluster_hm.png')

    print("Computing CNV UMAP and CNV score")
    cnv.tl.umap(adata)
    cnv.tl.cnv_score(adata)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))
    ax4.axis("off")
    cnv.pl.umap(
        adata,
        color="cnv_leiden",
        legend_loc="on data",
        legend_fontoutline=2,
        ax=ax1,
        show=False,
    )
    cnv.pl.umap(adata, color="cnv_score", ax=ax2, show=False)
    cnv.pl.umap(adata, color=celltype_key, ax=ax3, save='cnv_umap.png')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 11))
    cnv.pl.umap(
        adata,
        color="cnv_leiden",
        legend_loc="on data",
        legend_fontoutline=2,
        ax=ax1,
        show=False,
    )
    cnv.pl.umap(adata, color="cnv_score", ax=ax2, show=False)
    cnv.pl.umap(adata, color="batch", ax=ax3, show=False)
    cnv.pl.umap(adata, color="sample", ax=ax4, save='cnv_umap_batchandsample.png')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 12), gridspec_kw={"wspace": 0.6})
    ax4.axis("off")
    sc.pl.umap(adata, color="cnv_leiden", ax=ax1, show=False)
    sc.pl.umap(adata, color="cnv_score", ax=ax2, show=False)
    sc.pl.umap(adata, color=celltype_key, ax=ax3, save='regular_umap_withcnvscore.png')
    
    return adata

