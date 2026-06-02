#!/usr/bin/env python
# coding: utf-8

# # Wrapper functions for scanpy

# In[ ]:


import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata as ad
from pathlib import Path
import doubletdetection as dd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
from scipy.stats import median_abs_deviation


# In[ ]:


# version control
print("seaborn:", sns.__version__)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scanpy:", sc.__version__)
print("doubletdetection:", dd.__version__)
sns.set_theme()


# In[ ]:


def create_output_dir(master_dir, sub_dir_name, change_dir=False):
    '''
    Create an output directory as a subdirectory 'sub_dir_name' string within a parent directory master_dir
    Will not overwrite the files within that directory
    2025-05-28 moved to functions file
    2025-08-29 added change_dir
    '''
    output_dir = master_dir / sub_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Created output directory {output_dir}')
    if change_dir:
        os.chdir(output_dir)
        sc.settings.figdir = output_dir
        print(f'Default output directory changed to {output_dir}')
    return output_dir


# In[ ]:


def import_cr_outs(directory, verbose=True): 
    """
    This is designed for a cellranger multi output directory (tested cellranger 9.0.1 outputs)
    Function reads in all the filtered h5 files from cellranger output directory to separate anndata objects
    Then adds batch and sample names and filename and makes var_names unique.
    Importantly, it also adds the batch/sample name to each barcode, making each barcode unique to a given batch/sample.

    Parameters:
    directory (str): the cellranger multi output directory with the typical output structure

    Returns:
    a list of AnnData objects in the directory
    """
    directory = Path(directory)
    # navigate to the subdirectory where the per-sample output files are stored
    output_directory = Path(directory) / 'outs' / 'per_sample_outs'
    # get the list of subdirectories, which should represent the demultiplexed individual samples (e.g. 4 samples for on-chip multiplexing)
    subdirs = [item for item in output_directory.iterdir() if item.is_dir()]
    
    data_list = []
    
    for subdir in subdirs: # assuming there is one anndata per subdirectory
        # construct the filepath and load the anndata object
        filepath = subdir / 'count' / 'sample_filtered_feature_bc_matrix.h5'
        adata = sc.read_10x_h5(filepath, gex_only=True)

        if verbose:
            print(str(adata.n_obs) + " cells by " + str(adata.n_vars) + " genes")
        
        # Make variable names unique
        adata.var_names_make_unique()
        
        # Add metadata to the AnnData object
        # Assume the name of the subdirectory is the sample name
        sample = str(subdir).replace((str(output_directory) + '/'), '')
        adata.obs['sample'] = sample
        adata.uns['sample'] = sample
        adata.uns['filename'] = str(filepath)
        # Assume the name of the cellranger directory is the batch
        batch = directory.name
        adata.obs['batch'] = batch
        adata.uns['batch'] = batch

        if verbose:
            print(str(adata.uns['batch']) + " " + str(adata.uns['sample']))

        # Append the sample name to each barcode for a unique identifier
        adata.obs.index = adata.obs.index + "_" + sample
        
        # Append the annotated AnnData to the list
        data_list.append(adata)

    return data_list


# In[ ]:


def import_and_label_data(directory, label: str='batch', keyword=None): 
    """
    Function reads in all the h5 files from directory to separate anndata objects
    Then adds batch/sample name and filename and makes var_names unique.
    Importantly, it also adds the batch/sample name to each barcode, making each barcode unique to a given batch/sample.

    Parameters:
    directory (str): The path to the directory containing the .h5 files
    label (str): 'batch' or 'sample' or other label for the anndata object loaded from the h5 file

    Returns:
    a list of AnnData objects in the directory
    2025-12-2: added label argument for batch or sample and added file format for cellranger multi output
    2026-03-26: added keyword argument to selectively import only some h5 files
    """
    
    # List all h5 files in the specified directory
    if keyword:
        files = [f for f in os.listdir(directory) if (f.endswith('.h5') and keyword in f)]
    else:
        files = [f for f in os.listdir(directory) if f.endswith('.h5')]
        
    data_list = []

    for file in files:
        # Load the AnnData object
        path = os.path.join(directory, file)
        adata = sc.read_10x_h5(path, gex_only=True)
        
        # Make variable names unique
        adata.var_names_make_unique()

        # Extract sample information from the filename
        if '_cb_feature_bc_matrix_filtered.h5' in file: # if cellbender output
            adata_label = file.replace('_cb_feature_bc_matrix_filtered.h5', '')
        elif '_sample_filtered_feature_bc_matrix.h5' in file: # if cellranger multi output
            adata_label = file.replace('_sample_filtered_feature_bc_matrix.h5', '')
        elif '_filtered_feature_bc_matrix.h5' in file: # if cellranger count output
            adata_label = file.replace('_filtered_feature_bc_matrix.h5', '')

        # Add metadata to the AnnData object
        adata.obs[label] = adata_label
        adata.uns[label] = adata_label
        adata.uns['filename'] = file

        # Append batch or sample name to each barcode for a unique identifier
        adata.obs.index = adata.obs.index + "_" + adata_label
        
        # Append the annotated AnnData to the list
        data_list.append(adata)

    return data_list


# In[ ]:


def import_and_label_mtx(directory, label='batch'):
    '''
    Given the master directory, the function reads in all the mtx files to separate anndata objects
    Then adds batch name and filename and makes var_names unique.
    Importantly, it also adds the batch name to each barcode, making each barcode traceable to a given batch.
    2025-12-4 added label parameter
    '''

    # List all subdirectories in the specified directory
    subdirs = os.listdir(directory)
    data_list = []

    for subdir in subdirs:
        # Load the AnnData object
        path = Path(directory) / subdir
        adata = sc.read_10x_mtx(path, gex_only = True)
        
        # Make variable names unique
        adata.var_names_make_unique()

        # Assume the name of the subdirectory is the batch
        batch = subdir

        # Add metadata to the AnnData object
        adata.obs[label] = batch
        adata.uns[label] = batch
        adata.uns['filename'] = str(path)

        # Append batch name to each barcode for a unique identifier
        adata.obs.index = adata.obs.index + "_" + batch
        
        # Append the annotated AnnData to the list
        data_list.append(adata)

    return data_list


# In[ ]:


# Function reads in all the h5ad files from directory to separate anndata objects
def import_data(directory): 
    # List all h5ad files in the specified directory
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


def import_from_subdirs(directory): 
    '''
    Given the master directory, the function reads in all the h5ad files in the subdirectories to separate anndata objects
    Does not read in files located in the master directory alone - must be in a subdirectory.
    '''

    # List all subdirectories in the specified directory
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    data_list = []

    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        files = [f for f in os.listdir(subdir_path) if f.endswith('.h5ad')]
        for file in files:
            # Load the AnnData object
            path = os.path.join(subdir_path, file)
            adata = sc.read_h5ad(path)
            # Append the annotated AnnData to the list
            data_list.append(adata)

    return data_list


# In[ ]:


def qc_adata(adata, label='batch', species='human'):
    '''
    Calculates qc metrics and creates basic qc plots.
    Returns adata with adata.var and adata.obs modified
    2025-12-2 added label parameter
    2025-12-16 added species parameter
    '''
    print()
    batch = adata.uns[label]
    print(batch)
    print(f"Number of cells before quality control: {adata.n_obs}")

    if species=='human':
        # mitochondrial genes "MT-" for human
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        # ribosomal genes all capital for human
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))

    if species=='mouse':
        # mitochondrial genes "mt-" for mouse
        adata.var["mt"] = adata.var_names.str.startswith("mt-")
        # ribosomal genes first letter capital for mouse
        adata.var["ribo"] = adata.var_names.str.startswith(("Rps", "Rpl"))

    # To avoid NaN in QC metric calculation results, first filter out cells with no counts
    sc.pp.filter_cells(adata, min_counts=1)
    print(f"Number of cells after filtering out cells with no counts: {adata.n_obs}")
    
    # calculate qc metrics
    print("Calculating quality control metrics")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True, log1p=True)

    # Plot qc metrics
    n_genes_filename = f'{batch}_counts_by_genes.png'
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", save=n_genes_filename)
    pct_mt_filename = f'{batch}_counts_by_mt.png'
    sc.pl.scatter(adata, "total_counts", "pct_counts_mt", save=pct_mt_filename)
    pct_ribo_filename = f'{batch}_counts_by_ribo.png'
    sc.pl.scatter(adata, "total_counts", "pct_counts_ribo", save=pct_ribo_filename)
    histo_filename = f'{batch}_counts_histogram.png'
    sns.displot(adata.obs["total_counts"], bins=100, kde=False)
    plt.savefig(histo_filename)
    plt.close()
    jointplot_filename = f'{batch}_counts_by_genes_jointplot.png'
    sns.jointplot(data=adata.obs, x="log1p_total_counts", y="log1p_n_genes_by_counts", kind="hex",)
    plt.savefig(jointplot_filename)
    plt.close()
    
    return adata


# In[ ]:


# helper function that computes which cells are greater than the given number of MADs away from the median of the given metric
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (np.median(M) + nmads * median_abs_deviation(M) < M)
    return outlier


# In[ ]:


# helper function that returns the lower bound and upper bound for each metric based on nmads and returns these two values as a tuple
def nmad_threshold(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    lower_bound = np.median(M) - nmads * median_abs_deviation(M)
    upper_bound = np.median(M) + nmads * median_abs_deviation(M)
    return lower_bound, upper_bound


# In[ ]:


def filter_counts(adata, label='batch', method='standard', min_counts=None, max_counts=None, min_genes=None, 
                  nmads_counts=None, nmads_genes=None):
    '''
    With method='standard', filters out cells with less than min_counts and greater than max_counts
    With method='MAD', filters out cells with greater than nmads_counts away from the median and nmads_genes away from the median,
    with low_genes as backstop minimum gene cutoff
    2025-12-2 added label parameter
    '''
    batch = adata.uns[label]
    print(f"Number of cells before filtering for counts: {adata.n_obs}")
    
    if method == 'standard':
        print("Using standard method")
        # QC plots
        counts_violin_filename = f'{batch}_counts_violin.png'
        sc.pl.violin(adata, "log1p_total_counts", save=counts_violin_filename)
        genes_violin_filename = f'{batch}_genes_violin.png'
        sc.pl.violin(adata, "log1p_n_genes_by_counts", save=genes_violin_filename)

        # Filter for qc metrics using set thresholds
        if min_counts != None:
            sc.pp.filter_cells(adata, min_counts=min_counts)
            print(f"Number of cells after filtering out cells with very low counts: {adata.n_obs}")
        if max_counts != None:
            sc.pp.filter_cells(adata, max_counts=max_counts)
            print(f"Number of cells after filtering out cells with very high counts: {adata.n_obs}")
        if min_genes != None:
            sc.pp.filter_cells(adata, min_genes=min_genes)
            print(f"Number of cells after filtering out cells with very low number of genes: {adata.n_obs}")

    elif method == 'MAD':
        print("Using median absolute deviation method")
        # Compute the outliers
        if nmads_counts == None: print("Warning: you must provide nmads_counts for MAD method")
        if nmads_genes == None: print("Warning: you must provide nmads_genes for MAD method")
        adata.obs["outlier"] = (is_outlier(adata, "log1p_total_counts", nmads_counts) | is_outlier(adata, "log1p_n_genes_by_counts", nmads_genes))
        print(adata.obs.outlier.value_counts())

        # Get the lower and upper bound for each metric for plotting purposes
        counts_lb, counts_ub = nmad_threshold(adata, "log1p_total_counts", nmads_counts)
        print(f"Counts lower bound {counts_lb} and upper bound {counts_ub}")
        genes_lb, genes_ub = nmad_threshold(adata, "log1p_n_genes_by_counts", nmads_genes)
        print(f"Genes lower bound {genes_lb} and upper bound {genes_ub}")

        # QC plots
        # Scatter plot with log total counts on x axis and log genes by counts on y axis
        log_n_genes_filename = f'{batch}_log_counts_by_log_genes.png'
        sc.pl.scatter(adata, "log1p_total_counts", "log1p_n_genes_by_counts", show=False)
        plt.axvline(x=counts_lb, color='blue', linestyle='--',)
        plt.axvline(x=counts_ub, color='blue', linestyle='--',)
        plt.axhline(y=genes_lb, color='red', linestyle='--',)
        plt.axhline(y=genes_ub, color='red', linestyle='--',)
        plt.savefig(log_n_genes_filename)
        plt.close()
        # Violin plot of counts
        counts_violin_filename = f'{batch}_counts_violin.png'
        sc.pl.violin(adata, "log1p_total_counts", show=False)
        plt.axhline(y=counts_lb, color='red', linestyle='--',)
        plt.axhline(y=counts_ub, color='red', linestyle='--',)
        plt.savefig(counts_violin_filename)
        plt.close()
        # Violin plot of genes
        genes_violin_filename = f'{batch}_genes_violin.png'
        sc.pl.violin(adata, "log1p_n_genes_by_counts", show=False)
        plt.axhline(y=counts_lb, color='red', linestyle='--',)
        plt.axhline(y=counts_ub, color='red', linestyle='--',)
        plt.savefig(genes_violin_filename)
        plt.close()
        
        # Filter out the outliers
        print(f"Total number of cells: {adata.n_obs}")
        adata = adata[~adata.obs.outlier]
        print(f"Number of cells after filtering out cells with very high or very low counts and genes: {adata.n_obs}")
    
    else:
        print('Warning: method must be equal to standard or MAD.')

    return adata


# In[ ]:


def filter_mt(adata, label='batch', method='standard', nmads_mt=5, max_mito=10):
    '''
    With method='standard', filters out cells with greater than max_mito percentage of counts
    With method='MAD', filters out cells with greater than nmads_mt away from the median
    2025-12-2 added label parameter
    '''
    batch = adata.uns[label]
    print(f"Number of cells before filtering for percentage of mitochondrial genes: {adata.n_obs}")
    if method == 'standard':
        # Plot qc metrics
        pct_mt_violin_filename = f'{batch}_mt_violin.png'
        sc.pl.violin(adata, "pct_counts_mt", save=pct_mt_violin_filename)

        # Filter for qc metrics using set thresholds
        adata = adata[adata.obs['pct_counts_mt'] <= max_mito, :]
        print(f"Number of cells after filtering out cells with high pct_counts_mt: {adata.n_obs}")

    elif method == 'MAD':
        # Compute the outliers
        adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", nmads_mt)
        print(adata.obs.mt_outlier.value_counts())

        # get the lower and upper bound for each metric for plotting purposes
        mt_lb, mt_ub = nmad_threshold(adata, "pct_counts_mt", nmads_mt)
        print(f"Mitochondrial percentage lower bound {mt_lb} and upper bound {mt_ub}")

        # Violin plots
        pct_mt_violin_filename = f'{batch}_mt_violin.png'
        sc.pl.violin(adata, "pct_counts_mt", show=False)
        plt.axhline(y=mt_lb, color='red', linestyle='--',)
        plt.axhline(y=mt_ub, color='red', linestyle='--',)
        plt.savefig(pct_mt_violin_filename)
        plt.close()
        
        # Filter out the outliers
        adata = adata[~adata.obs.mt_outlier]
        print(f"Number of cells after filtering out cells with very high percentage of mitochondrial genes: {adata.n_obs}")
    
    else:
        print('Warning: method must be equal to standard or MAD.')

    return adata


# In[ ]:


def filter_ribo(adata, label='batch', method='standard', nmads_ribo=5, max_ribo=20):
    '''
    With method='standard', filters out cells with less than min_counts and greater than max_counts

    With method='MAD', filters out cells with greater than nmads_counts away from the median and nmads_genes away from the median,
    with low_genes as backstop minimum gene cutoff

    2025-12-2 added label parameter
    '''
    batch = adata.uns[label]
    print(f"Number of cells before filtering for percentage of ribosomal genes: {adata.n_obs}")
    
    if method == 'standard':
        # Plot qc metrics
        pct_ribo_violin_filename = f'{batch}_ribo_violin.png'
        sc.pl.violin(adata, "pct_counts_ribo", save=pct_ribo_violin_filename)

        # Filter for qc metrics using set thresholds
        adata = adata[adata.obs['pct_counts_ribo'] <= max_ribo, :]
        print(f"Number of cells after filtering out cells with high pct_counts_ribo: {adata.n_obs}")

    elif method == 'MAD':
        # Compute the outliers
        adata.obs["ribo_outlier"] = is_outlier(adata, "pct_counts_ribo", nmads_ribo)
        print(adata.obs.ribo_outlier.value_counts())

        # Get the lower and upper bound for each metric for plotting purposes
        ribo_lb, ribo_ub = nmad_threshold(adata, "pct_counts_ribo", nmads_ribo)
        print(f"Ribosomal percentage lower bound {ribo_lb} and upper bound {ribo_ub}")

        # Violin plot
        pct_ribo_violin_filename = f'{batch}_ribo_violin.png'
        sc.pl.violin(adata, "pct_counts_ribo", show=False)
        plt.axhline(y=ribo_lb, color='red', linestyle='--',)
        plt.axhline(y=ribo_ub, color='red', linestyle='--',)
        plt.savefig(pct_ribo_violin_filename)
        plt.close()
        
        # Filter out the outliers
        adata = adata[~adata.obs.ribo_outlier]
        print(f"Number of cells after filtering out cells with very high percentage of ribosomal genes: {adata.n_obs}")

    else:
        print('Warning: method must be equal to standard or MAD.')

    return adata


# In[ ]:


def deciteseq(adata):
    '''
    Removing TCR and BCR genes since these were amplified in some of the RetroBio TCR sequencing samples
    2025-07-16 updated to remove bug that removed 57 nonrelated genes with the same prefixes. Now only removes TCR and BCR genes.
    '''
    print(f"Number of cells before removing TCR and BCR genes: {adata.n_obs}")
    tcr_genes = ['TRAV', 'TRAJ', 'TRAC', 
                 'TRDV', 'TRDD', 'TRDJ', 'TRDC', 
                 'TRGV', 'TRGJ', 'TRGC', 
                 'TRBV', 'TRBD', 'TRBJ', 'TRBC',]
    bcr_genes = ['IGHV', 'IGHD', 'IGHJ', 'IGHC', 
                 'IGKV', 'IGKJ', 'IGKC', 
                 'IGLV', 'IGLJ', 'IGLC', 
                 'IGHA', 'IGHE', 'IGHD', 'IGHG']
    tcr_bcr_gene = tcr_genes + bcr_genes
    def is_tcr_bcr_gene(gene_name):
        return any(gene_name.startswith(prefix) or gene_name == 'IGHM' for prefix in tcr_bcr_gene)
    filtered_genes = [gene for gene in adata.var_names if not is_tcr_bcr_gene(gene)]
    adata = adata[:, filtered_genes]
    print(f"Number of cells after removing TCR and BCR genes: {adata.n_obs}")
    return adata


# In[ ]:


def dd_find_doublets(adata, label='batch', algorithm="louvain", n_iters=10, p_thresh=1e-16, voter_thresh=0.5):
    '''
    This function is a wrapper for DoubletDetection. It detects but does not remove doublets.
    2025-03-12 parameterized
    2025-12-2 added label parameter
    '''
    adata = adata.copy() # copy to avoid removing genes from actual anndata and avoid log normalizing the original adata
    batch = adata.uns[label]
    print(batch)
    
    # remove "empty" genes
    print(f"Number of genes before filtering for min_genes: {adata.n_vars}")
    sc.pp.filter_genes(adata, min_cells=1)
    print(f"Number of genes after filtering for min_genes: {adata.n_vars}")
    
    # doublet detection core function
    clf = dd.BoostClassifier(n_iters=n_iters, clustering_algorithm=algorithm, standard_scaling=True, pseudocount=0.1, n_jobs=-1)
    doublets = clf.fit(adata.X).predict(p_thresh=p_thresh, voter_thresh=voter_thresh)
    doublet_score = clf.doublet_score()
    adata.obs["doublet"] = doublets
    adata.obs["doublet_score"] = doublet_score
    
    f = dd.plot.convergence(clf, save=f'{batch}_convergence_test.pdf', show=True, p_thresh=p_thresh, voter_thresh=voter_thresh)
    f3 = dd.plot.threshold(clf, save=f'{batch}_threshold_test.pdf', show=True, p_step=6)
    plt.close()

    # plot doublets
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["doublet", "doublet_score"], save=f"{batch}_doublet_umap.pdf")
    sc.pl.violin(adata, "doublet_score", save=f"{batch}_doublet_violin_plot.pdf")
    plt.close()
    
    return adata


# In[ ]:


def import_markers(csv_path, output_type='dict'):
    '''
    Imports markers from a csv file, with column header as cell type, and creates a dictionary (default) or df
    '''
    if output_type=='dict':
        marker_genes_df = pd.read_csv(csv_path)
        marker_genes = {col: marker_genes_df[col].dropna().tolist() for col in marker_genes_df.columns}
        print(marker_genes)
        return marker_genes
    elif output_type=='df':
        marker_genes_df = pd.read_csv(csv_path)
        marker_genes = {col: marker_genes_df[col].dropna().tolist() for col in marker_genes_df.columns}
        all_genes = [gene for genes in marker_genes.values() for gene in genes]
        print(all_genes)
        return all_genes
    else:
        print("Error: output_type arg must be either 'dict' or 'df'.")


# In[ ]:


def sc_calc_umap(adata, label='batch', n_hvg=2000, min_dist=0.3):
    '''
    calculates HVGs and calculates umap for each sample
    scanpy only, not scvi
    2025-12-2 added label parameter
    '''
    batch = adata.uns[label]
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    print(f"Calculating HVGs for {batch}")
    # subset=False is important here because if True the setgenes dotplot in plot_sample will not always work
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_hvg, layer="counts", subset=False)
    hgenes_filename = f'{batch}_highly_var_genes.png'
    sc.pl.highly_variable_genes(adata, save=str(hgenes_filename))
    print(f"Finding principal components for {batch}")
    sc.tl.pca(adata)
    print(f"Finding nearest neighbors for {batch}")
    sc.pp.neighbors(adata)
    print(f"Calculating UMAP for {batch}")
    sc.tl.umap(adata, min_dist=min_dist)
    return adata


# In[ ]:


def featureplot(adata, markers, save="fp.png", **kwargs):
    '''
    Makes feature plots from the given list of markers for manual annotation
    2025-03-10 Removed resolution from this function
    2025-07-17 pass save argument (ending in .png or .pdf) directly. The filename is the markers dict key + save argument
    2025-07-18 use_raw parameter added
    2025-12-16 added a default for the save parameter
    2026-01-09 markers can be a dict or a list
    2026-03-11 changed to kwargs
    '''    
    if isinstance(markers, list):
        for marker in markers:
            filename = f'{marker}_{save}'
            sc.pl.umap(adata, 
                       color=marker, 
                       frameon=False, 
                       ncols=4, 
                       save=filename, 
                       **kwargs)
            
    if isinstance(markers, dict):
        for key, value in markers.items():
            filename = f'{key}_{save}'
            sc.pl.umap(adata, 
                       color=value, 
                       frameon=False, 
                       ncols=4, 
                       save=filename, 
                       **kwargs)


# In[ ]:


def sc_cluster_and_plot(adata, res: float, markers: dict, label='batch'):
    '''
    for the non-scvi samples
    umap must be calculated first
    1) performs leiden clustering
    2) performs differential gene expression calculation for each cluster relative to the other clusters
    3) generates dotplots, trackplots, matrix plots, heatmaps, violin plots, and umaps

    returns adata with the clustering
    
    2025-12-2 added label parameter
    '''
    adata = adata.copy() # to avoid issues with returning only a view of the anndata
    batch = adata.uns[label]
    str_res = str(res).replace('.', '_') # Format resolution to avoid dots in the filename
    leiden_key = "leiden" + str_res

    # Perform leiden clustering at the given resolution
    sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2, key_added=leiden_key)
    sc.tl.dendrogram(adata, groupby=leiden_key)
    # Plot umap at the given clustering resolution
    umap_filename = f'{batch}_{str_res}_res.png'
    sc.pl.umap(adata, color=leiden_key, save=str(umap_filename))

    # Calculate differentially expressed genes per cluster compared to other clusters
    print(f"Calculating DEGs for {batch}")
    sc.tl.rank_genes_groups(adata, groupby=leiden_key, method="wilcoxon")
    rank_genes_df = sc.get.rank_genes_groups_df(adata, group=None)
    rank_genes_csv = f'{batch}_{str_res}_rank_genes_groups.csv'
    rank_genes_df.to_csv(rank_genes_csv, index=False)

    # Also get top 100 degs
    

    #Dotplot of pre-specified markers
    set_genes_filename = f'{batch}_{str_res}_set_genes.png'
    sc.pl.dotplot(adata, markers, groupby=leiden_key, standard_scale="var", save=str(set_genes_filename))
    
    # Dotplot of top 5 DEGS in each cluster
    rank_genes_filename = f'{batch}_{str_res}_top_genes.png'
    sc.pl.rank_genes_groups_dotplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=str(rank_genes_filename))

    heat_filename = f'{batch}_{str_res}_top_genes.png'
    sc.pl.rank_genes_groups_heatmap(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=str(heat_filename))

    matrix_filename = f'{batch}_{str_res}_top_genes.png'
    sc.pl.rank_genes_groups_matrixplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=str(matrix_filename))

    tracks_filename = f'{batch}_{str_res}_top_genes.png'
    sc.pl.rank_genes_groups_tracksplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=str(tracks_filename))

    sv_filename = f'{batch}_{str_res}_top_genes.png'
    sc.pl.rank_genes_groups_stacked_violin(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=str(sv_filename))
      
    qc_filename = f'{batch}_qc.png'
    qc = ['batch', 'n_counts', 'n_genes_by_counts', 'log1p_n_genes_by_counts',
       'total_counts', 'log1p_total_counts', 'total_counts_mt',
       'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo',
       'log1p_total_counts_ribo', 'pct_counts_ribo', 'doublet', 'doublet_score']
    sc.pl.umap(adata, color=qc, ncols=4, save=str(qc_filename))

    return adata


# In[ ]:


def plot_umap(adata, resolution, neighbors_key=None, add_var=None):
    '''
    Make umaps of different resolutions
    Make sure leiden clustering is done before this
    Provide add_var argument if you want to plot some other obs value, such as celltype, on the umap

    2025-07-16 removed batch and sample plots from this function
    2025-07-17 made neighbors_keys optional
    '''
    str_res = str(resolution).replace('.', '_')
    leiden_key = "leiden" + str_res
    
    if add_var == None:

        # UMAP
        umap_filename = f'_{str_res}.png'
        sc.pl.umap(
            adata, neighbors_key=neighbors_key, color=leiden_key,
            frameon=False, ncols=1, save=umap_filename,)

        # UMAP with clusters labeled on the plot
        umap_filename = f'_{str_res}_labeled.png'
        sc.pl.umap(
            adata, neighbors_key=neighbors_key, color=leiden_key,
            frameon=False, ncols=1, legend_loc="on data", save=umap_filename,)
    
    elif isinstance(add_var, str):
        colors = [leiden_key] + [add_var]
        
        # UMAP
        umap_filename = f'_{str_res}.png'
        sc.pl.umap(
            adata, neighbors_key=neighbors_key, color=colors,
            frameon=False, ncols=1, save=umap_filename,)

        # UMAP with clusters labeled on the plot
        umap_filename = f'_{str_res}_labeled.png'
        sc.pl.umap(
            adata, neighbors_key=neighbors_key, color=colors,
            frameon=False, ncols=1, legend_loc="on data", save=umap_filename,)


# In[ ]:


def sc_degs(adata, res: float, plots=['dotplot', 'heatmap', 'matrixplot', 'tracksplot', 'stacked_violin'], use_rep='X_scVI'):
    '''
    This function is a wrapper for sc.tl.rank_genes_groups that should be used for calculating DEGS for adata.raw of anndata
    The default of sc.tl.rank_genes_groups is use_raw=True
    
    Uses scanpy differential expression
    UMAP and leiden clustering must be performed first
    1) performs differential gene expression calculation for each cluster relative to the other clusters
    2) generates dotplots, trackplots, matrix plots, heatmaps, violin plots, and umaps

    2025-04-07 updated to remove batch
    2025-07-17 updated to remove use_raw=True from sc.tl.rank_genes_groups().
    Also updated to use a try-except block to catch a ValueError if there is only one cluster
    2025-08-08 added plots parameter to specify which plots are created
    2025-08-27 added use_rep parameter
    2025-12-4 added a check if dendrogram has already been computed to avoid recomputing it
    2025-12-10 added top 100 degs and changed csv file names
    2025-12-16 fixed bug in file names
    2026-04-24 fixed error handling
    '''
    str_res = str(res).replace('.', '_') # Format resolution to avoid dots in the filename
    leiden_key = "leiden" + str_res
    try:
        # if dendrogram has not already been computed, compute the dendrogram for each cluster
        if f'dendrogram_{leiden_key}' not in adata.uns:
            sc.tl.dendrogram(adata, groupby=leiden_key, use_rep=use_rep)
    
        # Calculate differentially expressed genes per cluster compared to all other clusters using scanpy method (one vs all)
        print(f"Calculating scanpy DEGs for resolution {res}")
        sc.tl.rank_genes_groups(adata, groupby=leiden_key, method="wilcoxon")
        
        # saving the dataframe with all the degs
        de_df = sc.get.rank_genes_groups_df(adata, group=None)
        csv_path = f'{str_res}_sc_all_degs.csv'
        de_df.to_csv(csv_path, index=False)
       
        # filter for top 100 degs for each cluster before saving the dataframe
        cats = adata.obs[leiden_key].cat.categories
        top100 = pd.DataFrame()
        for c in cats:
            de_filt = de_df[de_df['group'] == c]
            de_filt = de_filt[de_filt['logfoldchanges'] > 0]
            de_filt = de_filt[de_filt['pvals_adj'] < 0.05]
            de_filt = de_filt.sort_values('scores', kind='mergesort', ascending=False)
            de_filt = de_filt.head(n=100)
            top100 = pd.concat([top100, de_filt], axis=0)
        # write degs df - the filtered one
        csv_path = f'sc_top100_degs_{str_res}.csv'
        top100.to_csv(csv_path)

        if 'dotplot' in plots:
            # Dotplot of top 5 DEGS in each cluster
            rank_genes_filename = f'{str_res}_sc_top_genes.png'
            sc.pl.rank_genes_groups_dotplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=rank_genes_filename)

        if 'heatmap' in plots:
            heat_filename = f'{str_res}_sc_top_genes.png'
            sc.pl.rank_genes_groups_heatmap(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=heat_filename)

        if 'matrixplot' in plots:
            matrix_filename = f'{str_res}_sc_top_genes.png'
            sc.pl.rank_genes_groups_matrixplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=matrix_filename)

        if 'tracksplot' in plots:
            tracks_filename = f'{str_res}_sc_top_genes.png'
            sc.pl.rank_genes_groups_tracksplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=tracks_filename)

        if 'stacked_violin' in plots:
            sv_filename = f'{str_res}_sc_top_genes.png'
            sc.pl.rank_genes_groups_stacked_violin(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=sv_filename)

    except Exception as e:
        print(e)
    
    return adata


# In[ ]:


def cluster_stats(adata, resolutions: list, rep: str='X_pca', scores = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']):
    '''
    calculates clustering statistics
    2025-03-19
    2025-07-14 updated to parameterize the location of the distance matrix
    2025-07-15 add option for which scores to compute
    2025-07-18 made 'X_pca' the default for rep parameter
    2025-07-18 fix bug when there are fewer than 2 clusters. Also put whole function in a try-except block.
    2025-09-30 Davies-Bouldin *100

    adata = anndata object
    rep = key of the representation array in adata e.g. 'X_scVI' or 'X_pca'
    resolutions = list of float resolutions, e.g. [0.1, 0.2, 0.5, 0.8, 1.0]

    Dependencies: numpy, pandas
    
    References:
    https://github.com/scverse/scanpy/issues/222d
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    '''
    try:
        # transform list of float resolutions to strings
        string_resolutions = []
        for res in resolutions:
            resn = 'leiden' + str(res).replace('.', '_')
            string_resolutions.append(resn)
    
        # remove any resolutions for which there are less than 2 clusters
        string_resolutions = [r for r in string_resolutions if len(np.unique(adata.obs[r])) >= 2]
        
        cluster_scores = pd.DataFrame(index=string_resolutions)
    
        if 'Silhouette' in scores:
            # compute silhouette score of all clusters
            from sklearn.metrics import silhouette_score
            print("Calculating silhouette scores")
            sil_scores = []
            for str_res in string_resolutions:
                sil = silhouette_score(adata.obsm[rep], adata.obs[str_res], metric='euclidean')
                print(str_res + " " + str(sil))
                sil_scores.append(sil)
            cluster_scores = cluster_scores.assign(silhouette = sil_scores)
    
        if 'Calinski-Harabasz' in scores:
            # compute Calinksi-Harabasz Score of all clusters
            from sklearn.metrics import calinski_harabasz_score
            print("Calculating Calinski-Harabasz scores")
            ch_scores = []
            for str_res in string_resolutions:
                ch = calinski_harabasz_score(adata.obsm[rep], adata.obs[str_res])
                print(str_res + " " + str(ch))
                ch_scores.append(ch)
            cluster_scores = cluster_scores.assign(calinski_harabasz = ch_scores)
    
        if 'Davies-Bouldin' in scores:
            # Compute Davies-Bouldin score of all clusters
            from sklearn.metrics import davies_bouldin_score
            print("Calculating Davies-Bouldin scores")
            db_scores = []
            for str_res in string_resolutions:
                db = davies_bouldin_score(adata.obsm[rep], adata.obs[str_res])
                print(str_res + " " + str(db))
                db_scores.append(db)
            cluster_scores = cluster_scores.assign(davies_bouldin = db_scores)
    
        # save the dataframe as a csv
        cluster_scores.to_csv('cluster_scores.csv', index = True)
        print(cluster_scores)
    
        # transform scores to make them easier to plot
        if 'Silhouette' in scores: cluster_scores[sil] = cluster_scores['silhouette'] * 10
        if 'Calinski-Harabasz' in scores: cluster_scores[ch] = cluster_scores['calinski_harabasz'] / 1000
        if 'Davies-Bouldin' in scores: cluster_scores[db] = cluster_scores['davies_bouldin'] * 100
        # plot cluster scores
        fig, ax = plt.subplots()
        if 'Silhouette' in scores: plt.plot(cluster_scores[sil], label='Silhouette Score * 10')
        if 'Calinski-Harabasz' in scores: plt.plot(cluster_scores[ch], label='Calinski-Harabasz Score / 1000')
        if 'Davies-Bouldin' in scores: plt.plot(cluster_scores[db], label='Davies-Bouldin Score * 100')
        plt.xlabel("Leiden clustering resolution")
        plt.ylabel("Clustering score")
        plt.title('Clustering Metrics')
        plt.legend()
        ax.grid(False)
        ax.set_facecolor('white')
        plt.savefig("clustering_metrics.png")
        plt.close()

    except ValueError as e:
        print(f"Error: There are likely less than 2 clusters for at least one given resolution. Details: {e}")

