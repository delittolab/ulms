#!/usr/bin/env python
# coding: utf-8

# # Wrapper functions for scvi

# In[ ]:


import os
import numpy as np
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
from scipy.stats import median_abs_deviation


# In[ ]:


def check_training(scvi_model, save='elbo_plot.png'):
    history = scvi_model.history
    if callable(history):
        history = history()
    train_test_results = history["elbo_train"]
    train_test_results["elbo_validation"] = history["elbo_validation"]
    train_test_results.plot()
    plt.savefig(save)
    plt.close()


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


def import_gene_list(directory, filetype: str=".txt"):
    '''
    Dependencies: Path
    Import a bunch of gene lists from a directory
    Label them by filename and return as a dictionary sorted by key in alphabetical order
    filetype: ending of each gene list filename in the directory e.g. .txt or .csv
    '''
    # List all files in the specified directory and sort by name
    directory = Path(directory)
    # files = [f for f in os.listdir(directory) if f.endswith(filetype)]
    files = [f for f in directory.glob('*' + filetype) if f.is_file()]
    files.sort(key=lambda x: x.name)
    gene_dict = {}

    for file in files:
        genes = [x.strip() for x in open(file)] # list of genes
        # Add another dictionary entry containing the new gene list
        gene_dict[file.name.replace(filetype, "")] = genes # key is filename and value is gene list

    return gene_dict


# In[ ]:


# Gene scoring
def score(adata, scores: dict, groupby: str, save=''):
    '''
    Must pass a labeled anndata that will be grouped by adata.obs[groupby]
    This functions calculates gene scores in the scores dictionary.
    It plots umaps of each gene score and then makes a matrix plot (pass the groupby argument) to summarize.
    
    2025-08-20 removed section and clustering resolution parameters. Added groupby parameter instead of leiden_key.
    2025-09-06 removed the filtering for tumor and added size and save to the umap
    '''
    
    for key, value in scores.items():
        print(f'Calculating gene score {key}')
        try: # score genes and make umaps of the gene score
            sc.tl.score_genes(adata, value, score_name=key)
            sc.pl.umap(adata, color=key, save=f'{save}_{key}.png', size=0.2)
            plt.close()
        except ValueError as e:
            print(f"Error in gene score calculation for {key}. Details: {e}")

    try:
        sc.pl.matrixplot(adata, var_names=list(scores.keys()), groupby=groupby, standard_scale='var', save=f'{save}_tumor.png')
    except Exception as e:
        print(f"Error in making matrix plot of gene scores. Details: {e}")


# In[ ]:


# Gene scoring
def score_tumor(adata, scores: dict, tumor_cat: str, groupby: str, save=''):
    '''
    Must pass a clustered anndata that has cells annotated as 'Tumor' in adata.obs[tumor_cat]
    This functions calculates tumor gene scores in the scores dictionary.
    It plots umaps of each gene score and then makes a matrix plot (pass the groupby argument) to summarize.
    
    2025-08-20 removed section and clustering resolution parameters. Added groupby parameter instead of leiden_key.
    '''
    # subset the anndata to tumor cells and only score these
    tumor = adata[adata.obs[tumor_cat] == 'Tumor'].copy()
    
    for key, value in scores.items():
        print(f'Calculating gene score {key}')
        try: # score genes and make umaps of the gene score
            sc.tl.score_genes(tumor, value, score_name=key)
            sc.pl.umap(tumor, color=key, save=f'{key}.png')
            plt.close()
        except ValueError as e:
            print(f"Error in gene score calculation for {key}. Details: {e}")

    try:
        sc.pl.matrixplot(tumor, var_names=list(scores.keys()), groupby=groupby, standard_scale='var', save=f'{save}_tumor.png')
    except ValueError as e:
        print(f"Error in making matrix plot of gene scores. Details: {e}")


# In[ ]:


def import_and_label_data(directory, label='batch'): 
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
    """
    
    # List all files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    data_list = []

    for file in files:
        # Load the AnnData object
        path = os.path.join(directory, file)
        adata = sc.read_10x_h5(path, gex_only = True)
        
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


def import_and_label_mtx(directory):
    '''
    Given the master directory, the function reads in all the mtx files to separate anndata objects
    Then adds batch name and filename and makes var_names unique.
    Importantly, it also adds the batch name to each barcode, making each barcode unique to a given batch.
    '''

    # List all subdirectories in the specified directory
    subdirs = os.listdir(directory)
    data_list = []

    for subdir in subdirs:
        # Load the AnnData object
            # Load the AnnData object
        path = Path(directory) / subdir
        adata = sc.read_10x_mtx(path, gex_only = True)
        
        # Make variable names unique
        adata.var_names_make_unique()

        # Extract batch information from the filename
        # Extract sample and batch information from the filename
        # Assuming filenames are in the format 'sample_batch.h5ad'
        batch = subdir

        # Add metadata to the AnnData object
        adata.obs['batch'] = batch
        adata.uns['batch'] = batch
        adata.uns['filename'] = str(path)

        # Append batch name to each barcode for a unique identifier
        adata.obs.index = adata.obs.index + "_" + batch
        
        # Append the annotated AnnData to the list
        data_list.append(adata)

    return data_list


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


def import_from_subdirs(directory, label=None): 
    '''
    Given the master directory, the function reads in all the h5ad files in the subdirectories to separate anndata objects
    Does not read in files located in the master directory alone - must be in a subdirectory.

    2025-08-14 Added optional label argument
    '''

    # List all subdirectories in the specified directory
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    data_list = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        files = [f for f in os.listdir(subdir_path) if f.endswith('.h5ad')]
        if label != None:
            files = [f for f in files if str(label) in f]
        for file in files:
            # Load the AnnData object
            path = os.path.join(subdir_path, file)
            adata = sc.read_h5ad(path)
            # Append the annotated AnnData to the list
            data_list.append(adata)

    return data_list


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


def scvi_degs(adata, model, resolution, set_genes, rep_key, norm_layer):
    '''
    make sure leiden clustering is done before this
    must get the "scvi_normalized" layer from the scvi model before using this

    2025-12-10 fixed Path bug and changed csv file names
    2025-12-16 removed output_dir
    '''
    str_res = str(resolution).replace('.', '_')
    leiden_key = "leiden" + str_res
    
    # computationally expensive
    de_df = model.differential_expression(groupby=leiden_key)
    
    # find top 5 markers of each leiden cluster
    markers = {}
    cats = adata.obs[leiden_key].cat.categories
    for c in cats:
        top_degs = de_df[de_df['group1'] == c]
        top_degs = top_degs[top_degs['lfc_mean'] > 0]
        top_degs = top_degs[top_degs['is_de_fdr_0.05'] == True]
        top_degs = top_degs.sort_values('proba_de', kind='mergesort', ascending=False)
        markers[c] = top_degs.index.tolist()[:5]
    
    # filter for top 100 degs for each cluster before saving the dataframe
    top100 = pd.DataFrame()
    for c in cats:
        de_filt = de_df[de_df['group1'] == c]
        de_filt = de_filt[de_filt['lfc_mean'] > 0]
        de_filt = de_filt[de_filt['is_de_fdr_0.05'] == True]
        de_filt = de_filt.sort_values('proba_de', kind='mergesort', ascending=False)
        de_filt = de_filt.head(n=100)
        top100 = pd.concat([top100, de_filt], axis=0)
    
    # write degs df - all of them
    csv_path = f'scvi_all_degs_{str_res}.csv'
    de_df.to_csv(csv_path)
    # write degs df - the filtered one
    csv_path = f'scvi_top100_degs_{str_res}.csv'
    top100.to_csv(csv_path)
    
    sc.tl.dendrogram(adata, groupby=leiden_key, use_rep=rep_key) # important to use the SCVI obsm rep here
    
    # dotplot of top genes
    tg_filename = f'scvi_dp_topgenes_{str_res}.png'
    sc.pl.dotplot(adata, markers, groupby=leiden_key, dendrogram=True,
                  color_map="Blues", swap_axes=True, use_raw=True, standard_scale="var", save=tg_filename)
    
    # dotplot of set genes from 'all_genes' csv file loaded earlier
    sg_filename = f'scvi_dp_setgenes_{str_res}.png'
    sc.pl.dotplot(adata, set_genes, groupby=leiden_key, dendrogram=False,
                  swap_axes=False, use_raw=True, standard_scale="var", save=sg_filename)
    
    # heatmap of top genes
    # make sure you have saved the normalized expression to this layer already
    hm_filename = f'scvi_hm_{str_res}.png'
    sc.pl.heatmap(adata, markers, groupby=leiden_key, layer=norm_layer, 
                  standard_scale="var", dendrogram=True, save=hm_filename)
    
    return adata


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


# In[ ]:


def median_cluster_qc(adata, resolution: float, qc_metrics: list=['n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_mt', 'log1p_total_counts_mt']):
    '''
    Smooth quality control metrics by calculating the median value across a given cluster and plotting it.
    resolution: a single float resolution
    qc_metrics: a list of qc metrics available in adata.obs
    '''
    adata = adata.copy()
    str_res = str(resolution).replace(".", "_")
    leiden_key = 'leiden' + str_res
    
    # calculate median of each QC metric
    for metric in qc_metrics:
        str_metric = 'median_' + metric
        adata.obs[str_metric] = adata.obs.groupby(leiden_key, observed=True)[metric].transform('median')

    sc.pl.umap(adata, color=qc_metrics, frameon=False, save=f'qc_median_{str_res}.png')

    for metric in qc_metrics:
        str_metric = 'median_' + metric
        # make a dataframe of the median value of the metric in each leiden cluster
        median_metric = adata.obs.groupby(leiden_key, observed=True)[str_metric].unique().reset_index().astype(float)
        median_metric[leiden_key] = median_metric[leiden_key].astype(int)

        # Create a bar plot for each metric
        plt.figure(figsize=(12, 6))
        sns.barplot(x=leiden_key, y=str_metric, data=median_metric)
        plt.title(f'Median {metric} for each cluster')
        plt.ylabel(f'Median {metric}')
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(f'{str_metric}_barplot.png')
        plt.close()

        # Create a box plot for each metric
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=str_metric, data=median_metric)
        plt.title(f'Median {metric} for each cluster')
        plt.ylabel(f'Median {metric}')
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(f'{str_metric}_boxplot.png')
        plt.close()


# In[ ]:


# def scviva_degs(adata, model, resolution, set_genes, rep_key, norm_layer):
#     '''
#     make sure leiden clustering is done before this
#     must get the "scvi_normalized" layer from the scvi model before using this

#     '''
#     str_res = str(resolution).replace('.', '_')
#     leiden_key = "leiden" + str_res
    
#     # computationally expensive
#     de_df = model.differential_expression(groupby=leiden_key)
    
#     # find top 5 markers of each leiden cluster
#     markers = {}
#     cats = adata.obs[leiden_key].cat.categories
#     for c in cats:
#         top_degs = de_df[de_df['group1'] == c]
#         top_degs = top_degs[top_degs['lfc_mean'] > 0]
#         top_degs = top_degs[top_degs['is_de_fdr_0.05'] == True]
#         top_degs = top_degs.sort_values('proba_de', kind='mergesort', ascending=False)
#         markers[c] = top_degs.index.tolist()[:5]
    
#     # filter for top 100 degs for each cluster before saving the dataframe
#     top100 = pd.DataFrame()
#     for c in cats:
#         de_filt = de_df[de_df['group1'] == c]
#         de_filt = de_filt[de_filt['lfc_mean'] > 0]
#         de_filt = de_filt[de_filt['is_de_fdr_0.05'] == True]
#         de_filt = de_filt.sort_values('proba_de', kind='mergesort', ascending=False)
#         de_filt = de_filt.head(n=100)
#         top100 = pd.concat([top100, de_filt], axis=0)
    
#     # write degs df - all of them
#     csv_path = f'scvi_all_degs_{str_res}.csv'
#     de_df.to_csv(csv_path)
#     # write degs df - the filtered one
#     csv_path = f'scvi_top100_degs_{str_res}.csv'
#     top100.to_csv(csv_path)
    
#     sc.tl.dendrogram(adata, groupby=leiden_key, use_rep=rep_key) # important to use the SCVI obsm rep here
    
#     # dotplot of top genes
#     tg_filename = f'scvi_dp_topgenes_{str_res}.png'
#     sc.pl.dotplot(adata, markers, groupby=leiden_key, dendrogram=True,
#                   color_map="Blues", swap_axes=True, use_raw=True, standard_scale="var", save=tg_filename)
    
#     # dotplot of set genes from 'all_genes' csv file loaded earlier
#     sg_filename = f'scvi_dp_setgenes_{str_res}.png'
#     sc.pl.dotplot(adata, set_genes, groupby=leiden_key, dendrogram=False,
#                   swap_axes=False, use_raw=True, standard_scale="var", save=sg_filename)
    
#     # heatmap of top genes
#     # make sure you have saved the normalized expression to this layer already
#     hm_filename = f'scvi_hm_{str_res}.png'
#     sc.pl.heatmap(adata, markers, groupby=leiden_key, layer=norm_layer, 
#                   standard_scale="var", dendrogram=True, save=hm_filename)
    
#     return adata

