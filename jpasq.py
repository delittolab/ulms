#!/usr/bin/env python
# coding: utf-8

# # Functions for G4X analysis with squidpy

# In[ ]:


import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import anndata as ad
from tifffile import imread
import spatialdata as sd

plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator


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


# Reload list of anndatas from saved h5 anndata objects.
def import_data(directory): 
    # List all files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('.h5ad')]
    data_list = []

    for file in files:
        # Load the AnnData object
        path = os.path.join(directory, file)
        adata = sc.read_h5ad(path)
        # Append the AnnData to the list
        data_list.append(adata)

    return data_list


# In[ ]:


def load_g4x(input_path):
    '''
    Load a single G4X file
    '''
    input_path = Path(input_path)
    adata = sc.read_h5ad(input_path)
    
    # reformatting: transfer cell ids into obs_names and gene ids into var_names
    adata.obs_names = [i for i in adata.obs['cell_id']]
    adata.var_names = [i for i in adata.var['gene_id']]
    
    # removing negative control probes and negative control sequences
    NCP_index = adata.var['gene_id'][adata.var['gene_id'].str.startswith("NCP")]
    NCS_index = adata.var['gene_id'][adata.var['gene_id'].str.startswith("NCS")]
    adata = adata[:, ~adata.var_names.isin(NCP_index)]
    adata = adata[:, ~adata.var_names.isin(NCS_index)]
    
    # Store the spatial coordinate data in obsm for squidpy
    adata.obsm['spatial'] = adata.obs[['cell_x', 'cell_y']].copy().to_numpy()

    # Store the section label in adata.obs and adata.uns for easy access
    section = input_path.name.replace('_feature_matrix.h5', '')
    adata.obs['section'] = section
    adata.uns['section'] = section

    return adata


# In[ ]:


def load_all_g4x(directory, recursive=False):
    '''
    load all G4X h5 files in a given directory (not recursive)
    returns a list of anndata objects

    If recursive = False, assumes all h5 files are gathered in a single directory
    If recursive = True, assumes the default G4X output file structure

    2025-10-15 added recursive option
    '''
    directory = Path(directory)
    
    if not recursive:
        # List all h5 files in the specified directory using Path
        files = [f for f in directory.iterdir() if f.is_file() and f.suffix == '.h5']

    elif recursive:
        # recursively list all h5 files in the specified master directory
        files = list(directory.rglob("*.h5"))

    data_list = []
    for file in files:
        # Load the AnnData object
        adata = load_g4x(file)
        # Append the annotated AnnData to the list
        data_list.append(adata)
    
    return data_list


# In[ ]:


def load_g4x_zarr(directory, section: str, proseg_folder_label: str, multiome=False, channel_names=None):
    '''
    For proseg outputs
    Loads a spatialdata object from a .zarr file (directory) in the typical proseg output structure

    2025-11-10 added multiome argument to load in the multichannel protein image as well
    '''
    def get_proseg_path(directory, section: str, proseg_folder_label=proseg_folder_label):
        '''
        helper function that returns the path to the G4X proseg output file
        directory is the raw data directory with the typical G4X structure.
        In the segmentation subdirectory, there is a proseg_output subdirectory that contains the zarr file
        '''
        directory = Path(directory)
        subdirs = [item for item in directory.iterdir() if item.is_dir()]
        # get the directory of the lane
        if '01' in section:
            [lane_dir] = [dir for dir in subdirs if 'L001' in str(dir)]
        elif '02' in section:
            [lane_dir] = [dir for dir in subdirs if 'L002' in str(dir)]
        elif '03' in section:
            [lane_dir] = [dir for dir in subdirs if 'L003' in str(dir)]
        elif '04' in section:
            [lane_dir] = [dir for dir in subdirs if 'L004' in str(dir)]
    
        zarr_path = lane_dir / f'{section}/segmentation/{proseg_folder_label}/proseg-output.zarr'
        return zarr_path

    def get_img_path(directory, section: str, img_type='HE'):
        '''
        helper function that returns the path to the G4X H&E image
        directory is the raw data directory with the typical G4X structure.
        '''
        directory = Path(directory)
        subdirs = [item for item in directory.iterdir() if item.is_dir()]
        # get the directory of the lane
        if '01' in section:
            [lane_dir] = [dir for dir in subdirs if 'L001' in str(dir)]
        elif '02' in section:
            [lane_dir] = [dir for dir in subdirs if 'L002' in str(dir)]
        elif '03' in section:
            [lane_dir] = [dir for dir in subdirs if 'L003' in str(dir)]
        elif '04' in section:
            [lane_dir] = [dir for dir in subdirs if 'L004' in str(dir)]

        if img_type=='HE':
            img_path = lane_dir / f'{section}/g4x_viewer/{section}_HE.ome.tiff'
        elif img_type=='multiome':
            img_path = lane_dir / f'{section}/g4x_viewer/{section}.ome.tiff'
        else: 
            raise ValueError("img_type must be either 'HE' or 'multiome'.")
        return img_path
    
    # read in the proseg output as a spatialdata object
    sdata = sd.read_zarr(get_proseg_path(directory, section))
    
    # access the anndata stored in the spatialdata object
    adata = sdata.tables["table"]
    
    # add the section ID
    adata.obs["section"] = section
    adata.uns["section"] = section
    
    # add the section ID to the cell indices
    adata.obs["cell_name"] = adata.obs["cell"].astype(str).radd(f"{section}_")
    adata.obs_names = adata.obs["cell_name"].astype(str)

    # read in the image and store in sdata
    if not multiome:
        hne = imread(get_img_path(directory, section, img_type='HE'))
        hne = sd.models.Image2DModel.parse(hne, dims=['y', 'x', 'c'],)
        sdata.images = {'hne' : hne}

    elif multiome:
        hne = imread(get_img_path(directory, section, img_type='HE'))
        hne = sd.models.Image2DModel.parse(hne, dims=['y', 'x', 'c'],)
        img = imread(get_img_path(directory, section, img_type='multiome'))
        img = sd.models.Image2DModel.parse(img, dims=['c', 'y', 'x'], c_coords=channel_names)
        sdata.images = {'hne': hne, 'multiome' : img}
    
    return sdata


# In[ ]:


def connect_adata_to_sdata(adata, sdata):
    '''
    Adds or replaces the table of a spatialdata object with a new adata
    sdata: the existing spatialdata object
    adata: the new anndata that you want to put into your spatialdata object instead of the one that is already there

    Connects the new anndata to the segmentation mask and H&E image
    2025-11-25 scale points as well
    '''
    # replace the anndata with the annotated anndata
    sdata.tables['table'] = adata
    
    # subset the segmentation mask to the same cells
    segs = sdata.shapes['cell_boundaries']
    segs = segs[segs['cell'].isin(adata.obs['cell'])]
    sdata.shapes['cell_boundaries'] = segs
    
    # Connect the anndata to the other spatial elements
    # https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/sdata_from_scratch.html#prepare-and-connect-the-anndata-to-the-rest-of-the-data
    sdata["table"].obs["region"] = "cell_boundaries"
    sdata.set_table_annotates_spatialelement(
        table_name="table", 
        region="cell_boundaries", 
        region_key="region", 
        instance_key="cell" # this matches the column adata.obs['cell'] with the column sdata.shapes['cell_boundaries']['cell']
    )
    
    # scale the shapes to the image coordinate system
    scale_factor = 1 / 0.3125 # Per Singular 0.3125 pixel = 1 um
    scale = sd.transformations.Scale([scale_factor, scale_factor], axes=("y", "x"))
    sd.transformations.set_transformation(sdata.shapes["cell_boundaries"], scale, to_coordinate_system="global")
    sd.transformations.set_transformation(sdata.points["transcripts"], scale, to_coordinate_system="global")
    
    return sdata


# In[ ]:


def g4x_qc(sdata=None, adata=None, min_counts=20, min_genes=5, segmentation='proseg', min_size_pct=0.5, max_size_pct=99.5):
    '''
    This works on a proseg spatialdata object or cellpose anndata object.
    2025-10-15 added input argument.
    2025-10-28 parameterized the size percentile cutoffs
    '''
    if segmentation == 'proseg':
        adata = sdata.tables['table']
        section = adata.uns['section']
        
        # calculate QC metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
        
        # spatial plots before QC
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=['log1p_total_counts', 'log1p_n_genes_by_counts'], save=f'{section}_beforeqc.png')
        plt.close()
        sdata.pl.render_shapes("cell_boundaries", color='log1p_total_counts').pl.show(save=f'{section}_seg_beforeqc.png')
        plt.close()

        # QC histograms
        g4x_qc_plot(adata, segmentation=segmentation)
        
        # QC for cell counts and genes
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        # QC for cell size (Proseg gives spatial outliers huge cell sizes)
        lo, hi = np.percentile(adata.obs["surface_area"].to_numpy(), [min_size_pct, max_size_pct])
        adata = adata[(adata.obs["surface_area"] >= lo) & (adata.obs["surface_area"] <= hi)].copy()
        sdata.tables['table'] = adata # must put adata back into sdata
        
        # subset the segmentation mask to the same cells
        segs = sdata.shapes['cell_boundaries']
        segs = segs[segs['cell'].isin(adata.obs['cell'])]
        sdata.shapes['cell_boundaries'] = segs
        
        # spatial plots after QC
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=['log1p_total_counts', 'log1p_n_genes_by_counts'], save=f'{section}_afterqc.png')
        plt.close()
        sdata.pl.render_shapes("cell_boundaries", color='log1p_total_counts').pl.show(save=f'{section}_seg_afterqc.png')
        plt.close()
        
        return sdata

    elif segmentation == 'cellpose':
        '''
        Takes a typical G4X cellpose anndata as input and assumes QC metrics have already been calculated
        '''
        section = adata.uns['section']
        
        # spatial plots before QC
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=['log1p_total_counts', 'log1p_n_genes_by_counts'], save=f'{section}_beforeqc.png')
        plt.close()

        # QC histograms
        g4x_qc_plot(adata, segmentation=segmentation)
        
        # QC for cell counts and genes
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        # spatial plots after QC
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=['log1p_total_counts', 'log1p_n_genes_by_counts'], save=f'{section}_afterqc.png')
        plt.close()
        
        return adata


# In[ ]:


def g4x_qc_plot(adata, segmentation='cellpose'):
    '''
    Makes G4X QC plots for a single anndata object
    2025-10-28 added proseg option
    '''
    section = adata.uns['section']

    if segmentation == 'cellpose':
        total_count_hist = sns.histplot(adata.obs['total_counts'], kde=False,)
        total_count_hist.set_title('Total transcripts per cell')
        plt.savefig(f'{section}_total_counts.png')
        plt.close()
    
        unique_transcripts_hist = sns.histplot(adata.obs['n_genes_by_counts'], kde=False)
        unique_transcripts_hist.set_title('Unique transcripts per cell')
        plt.savefig(f'{section}_n_genes_by_counts.png')
        plt.close()
    
        cell_area_hist = sns.histplot(adata.obs['nuclei_expanded_area'], kde=False)
        cell_area_hist.set_title('Segmented Cell Area')
        plt.savefig(f'{section}_nuclei_expanded_area.png')
        plt.close()
    
        nucleus_ratio_hist = sns.histplot(adata.obs['nuclei_area']/adata.obs['nuclei_expanded_area'], kde=False)
        nucleus_ratio_hist.set_title('Nucleus ratio')
        plt.savefig(f'{section}_nucleus_ratio.png')
        plt.close()
    elif segmentation == 'proseg':
        total_count_hist = sns.histplot(adata.obs['log1p_total_counts'], kde=False,)
        total_count_hist.set_title('Total transcripts per cell (log1p)')
        plt.savefig(f'{section}_total_counts.png')
        plt.close()
    
        unique_transcripts_hist = sns.histplot(adata.obs['log1p_n_genes_by_counts'], kde=False)
        unique_transcripts_hist.set_title('Unique transcripts per cell (log1p)')
        plt.savefig(f'{section}_n_genes_by_counts.png')
        plt.close()
    
        cell_area_hist = sns.histplot(adata.obs['surface_area'], kde=False)
        cell_area_hist.set_title('Segmented Cell Area')
        plt.savefig(f'{section}_surface_area.png')
        plt.close()
    
        cell_volume_hist = sns.histplot(adata.obs['volume'], kde=False)
        cell_volume_hist.set_title('Segmented Cell Volume')
        plt.savefig(f'{section}_volume.png')
        plt.close()
    else:
        print('ERROR: segmentation must be specified as either cellpose or proseg')


# In[ ]:


def find_spatial_outliers(adata, n_neighbors: int=30, method='max', n_std: int=10):
    '''
    Finds but does not remove spatial outliers: cells that are located far away from other cells.

    adata = anndata object
        - must have the section id stored in adata.uns['section']
        - must have the spatial coordinates stored in adata.obsm['spatial']
    n_neighbors = number of neighbors for use in the KNN neighborhood calculation
    If method = max, use the maximum distance from other cells in the neighborhood for delineating outliers.
    If method = min, use the minimum nonzero distance from other cells in the neighborhood for delineating outliers.
    If method = mean, use the mean distance from other cells in the neighborhood for delineating outliers.
    n_std = number of standard deviations from the median above which a cell will be defined as an outlier.

    Returns a column of adata.obs: True if outlier, False if not an outlier

    Dependencies:
        - import scanpy as sc
        - import squidpy as sq
        - import numpy as np
        - import matplotlib.pyplot as plt
    '''
    adata = adata.copy()
    section = adata.uns['section']
    # Use knn to calculate the euclidean distance to the given number of neighbors
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=n_neighbors, key_added='spatial')
    distances = adata.obsp['spatial_distances']

    if method == 'max': # first calculate the maximum distances of each cell to its nearest neighbors
        max_distances = distances.max(axis=1).toarray()
        spatial_dist_hist = sns.histplot(max_distances, kde=False)
        spatial_dist_hist.set_title('Maximum KNN Spatial Distances')
        plt.savefig(f'{section}_max_knn_spatial_distances.png') # histogram of distances will likely have a long tail
        plt.close()
        
        median_max_distance = np.median(max_distances)
        std_max_distance = np.std(max_distances)

        adata.obs['max_distances'] = max_distances
        adata.obs['spatial_outlier'] = max_distances > median_max_distance + n_std*std_max_distance
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=["max_distances", "spatial_outlier"], 
                              wspace=0.2, save=f'{section}_max_dist_outliers.png')
        plt.close()
        
    elif method == 'min': # first calculate the minimum nonzero distances of each cell to its nearest neighbors
        min_distances = []
        for i in range(distances.shape[0]):
            row = distances.getrow(i).toarray() # iterate row by row
            masked_row = np.ma.masked_values(row, 0.0, copy=False) # remove the zeroes by creating a masked array without copying the original array
            row_min = masked_row.min()
            min_distances.append(row_min)
        
        spatial_dist_hist = sns.histplot(min_distances, kde=False)
        spatial_dist_hist.set_title('Minimum KNN Spatial Distances')
        plt.savefig(f'{section}_min_knn_spatial_distances.png') # histogram of distances will likely have a long tail
        plt.close()
        
        median_min_distance = np.median(min_distances)
        std_min_distance = np.std(min_distances)

        adata.obs['min_distances'] = min_distances
        adata.obs['spatial_outlier'] = min_distances > median_min_distance + n_std*std_min_distance
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=["min_distances", "spatial_outlier"], 
                              wspace=0.2, save=f'{section}_min_dist_outliers.png')
        plt.close()

    elif method == 'mean': # first calculate the mean distances of each cell to its nearest neighbors
        mean_distances = distances.mean(axis=1).A1 # convert matrix to numpy array
        spatial_dist_hist = sns.histplot(mean_distances, kde=False)
        spatial_dist_hist.set_title('Mean KNN Spatial Distances')
        plt.savefig(f'{section}_mean_knn_spatial_distances.png') # histogram of distances will likely have a long tail
        plt.close()
        
        median_mean_distance = np.median(mean_distances)
        std_mean_distance = np.std(mean_distances)

        adata.obs['mean_distances'] = mean_distances
        adata.obs['spatial_outlier'] = mean_distances > median_mean_distance + n_std*std_mean_distance
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=["mean_distances", "spatial_outlier"], 
                              wspace=0.2, save=f'{section}_mean_dist_outliers.png')
        plt.close()
        
    else:
        raise ValueError("Available methods are 'max', 'min', or 'mean'.")

    return adata.obs['spatial_outlier']


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


def featureplot(adata, markers: dict, save: str, use_raw=True, neighbors_key=None):
    '''
    Makes feature plots from the given list of markers for manual annotation
    2025-03-10 Removed resolution from this function
    2025-07-17 pass save argument directly
    2025-07-18 use_raw parameter added
    '''
    for key, value in markers.items():
        filename = f'{key}_{save}'
        sc.pl.umap(adata, neighbors_key=neighbors_key, color=value, use_raw=use_raw, frameon=False, ncols=4, save=filename)


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


def sc_degs(adata, res: float, use_rep='X_pca', plots=['dotplot', 'heatmap', 'matrixplot', 'tracksplot', 'stacked_violin']):
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
    2025-08-26 added use_rep
    '''
    str_res = str(res).replace('.', '_') # Format resolution to avoid dots in the filename
    leiden_key = "leiden" + str_res
    try:    
        sc.tl.dendrogram(adata, groupby=leiden_key, use_rep=use_rep)
    
        # Calculate differentially expressed genes per cluster compared to other clusters using scanpy method
        print(f"Calculating scanpy DEGs for resolution {res}")
        sc.tl.rank_genes_groups(adata, groupby=leiden_key, method="wilcoxon")
        rank_genes_df = sc.get.rank_genes_groups_df(adata, group=None)
        rank_genes_csv = f'{str_res}_scanpy_rank_genes_groups.csv'
        rank_genes_df.to_csv(rank_genes_csv, index=False)

        if 'dotplot' in plots:
            # Dotplot of top 5 DEGS in each cluster
            rank_genes_filename = f'{str_res}_scanpy_top_genes.png'
            sc.pl.rank_genes_groups_dotplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=rank_genes_filename)

        if 'heatmap' in plots:
            heat_filename = f'{str_res}_scanpy_top_genes.png'
            sc.pl.rank_genes_groups_heatmap(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=heat_filename)

        if 'matrixplot' in plots:
            matrix_filename = f'{str_res}_scanpy_top_genes.png'
            sc.pl.rank_genes_groups_matrixplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=matrix_filename)

        if 'tracksplot' in plots:
            tracks_filename = f'{str_res}_scanpy_top_genes.png'
            sc.pl.rank_genes_groups_tracksplot(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=tracks_filename)

        if 'stacked_violin' in plots:
            sv_filename = f'{str_res}_scanpy_top_genes.png'
            sc.pl.rank_genes_groups_stacked_violin(adata, groupby=leiden_key, standard_scale="var", n_genes=5, save=sv_filename)

    except ValueError:
        print("There is a ValueError, likely from having fewer than 2 clusters at this resolution. No plots will be generated.")
    
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
        if 'Davies-Bouldin' in scores: cluster_scores[db] = cluster_scores['davies_bouldin'] * 10
        # plot cluster scores
        fig, ax = plt.subplots()
        if 'Silhouette' in scores: plt.plot(cluster_scores[sil], label='Silhouette Score * 10')
        if 'Calinski-Harabasz' in scores: plt.plot(cluster_scores[ch], label='Calinski-Harabasz Score / 1000')
        if 'Davies-Bouldin' in scores: plt.plot(cluster_scores[db], label='Davies-Bouldin Score * 10')
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


def get_adata(adata_list, section: str):
    '''
    Pulls the anndata with the given section ID out of the list of anndatas
    '''
    adata_list = [x for x in adata_list if x.uns['section'] == section]
    if len(adata_list) == 1: 
        return adata_list[0]
    else:
        print("There are multiple anndatas with the same section ID.")
        return None

def make_leiden_map(adata, resolution):
    '''
    Prints out the mapping dictionary for the clusters at the given resolution.
    Then you can easily copy it into the cell instead of typing it out.
    '''
    str_res = str(resolution).replace(".", "_")
    leiden_key = "leiden" + str_res
    cats = adata.obs[leiden_key].cat.categories.to_list()
    for cat in cats:
        print(f"'{cat}' : '',")

def help_annotate(adata_list, section, resolution):
    '''
    Call this function before annotating. It will:
    1) print the section ID and the anndata
    2) change the output directory to a new subdirectory named by the section
    3) print out the mapping dictionary for easy copy/paste
    '''
    adata = get_adata(adata_list, section)
    print(adata.uns['section'])
    print(adata)
    
    output_dir = jpasq.create_output_dir(output_master_dir, section)
    print(output_dir)
    os.chdir(output_dir)
    sc.settings.figdir = output_dir
    
    make_leiden_map(adata, resolution)
    return adata


# In[ ]:


def annotate(adata, section: str, resolution: float, leiden_map: dict, save=True):
    '''
    Call this after filling out the mapping dictionary. It will:
    1) Check that the number of clusters matches the number of dictionary entries 
    2) Add the celltype column to adata using the given mapping at that resolution
    3) Plot the annotated UMAP and spatial map
    4) Save the annotated adata as an h5ad file
    '''
    str_res = str(resolution).replace(".", "_")
    leiden_key = "leiden" + str_res

    # check by length
    if len(leiden_map) != len(np.unique(adata.obs[leiden_key])):
        print("There is a discrepancy in the number of categories that were annotated.")
    else: 
        print("Looks good!")
    
    # annotate
    adata.obs['celltype'] = adata.obs[leiden_key].map(leiden_map)

    # check by plotting
    sc.pl.umap(adata, color=[leiden_key, 'celltype'], ncols=1, save=f'{section}_annotated.png')
    sq.pl.spatial_scatter(adata, library_id='spatial', shape=None, color='celltype', save=f'{section}_annotated_spatial.png')
    
    # save
    if save==True: adata.write_h5ad(f'{section}_annotated.h5ad')


# In[ ]:


# Gene scoring
def score_tumor(adata, section: str, res: float, scores: dict):
    '''
    Must pass a clustered anndata that has cells annotated as 'Tumor' in adata.obs['celltype']
    This functions calculates tumor gene scores in the scores dictionary.
    It plots umaps of each gene score and then makes a matrix plot to summarize.
    '''
    str_res = str(res).replace(".", "_")
    leiden_key = "leiden" + str_res
    # subset the anndata to tumor cells and only score these
    tumor = adata[adata.obs['celltype'] == 'Tumor'].copy()
    
    for key, value in scores.items():
        print(f'Calculating gene score {key} in section {section}')
        try: # score genes and make umaps of the gene score
            sc.tl.score_genes(tumor, value, score_name=key)
            sc.pl.umap(tumor, color=key, save=f'{section}_{key}.png')
            plt.close()
        except ValueError as e:
            print(f"Error in gene score calculation for {key} in {section}. Details: {e}")

    try:
        sc.pl.matrixplot(tumor, var_names=list(scores.keys()), groupby=leiden_key, standard_scale='group')
    except ValueError as e:
        print(f"Error in making matrix plot of gene scores in {section}. Details: {e}")


# In[ ]:


def analyze(adata, 
            markers: dict,
            n_pcs: int=50, 
            n_neighbors: int=30, 
            resolutions: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
            min_dist: float=0.5,
           ):
    '''
    Sample-by-sample G4X workflow. Intended to be the full workflow after preprocessing each sample.

    Dependencies: scanpy, squidpy
    Also calls functions sc_degs(), plot_umap(), cluster_stats(), featureplot()

    adata: anndata object
    n_pcs: number of principal components for PCA
    n_neighbors: number of neighbors for sc.pp.neighbors()
    resolutions: list of resolutions at which to cluster
    min_dist: min_dist for umap calculation. Larger values will provide more global structure at the expense of local structure
    '''
    
    adata = adata.copy()
    
    # get section information
    section = adata.uns['section']
    print(section)
    
    adata.layers["counts"] = adata.X.copy() # save raw counts
    sc.pp.normalize_total(adata, inplace=True) # normalize
    sc.pp.log1p(adata) # log transform
    adata.raw = adata.copy() # save log transformed counts
    sc.pp.scale(adata, max_value=10) # scale to unit variance and zero mean and clip any gene expression above 10 std deviations

    # Dimension reduction
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pl.pca_variance_ratio(adata, n_pcs=n_pcs, save=f'{section}_pca_variance_ratio.png') # reduced from default given elbo plots and likely lower dimensionality of this data
    sc.pl.pca_loadings(adata, components = '1,2,3,4', save=f'{section}_pca_loadings.png')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors) # can increase from n_neighbors from default 15 for better global visualization
    sc.tl.umap(adata, min_dist=min_dist) # can increase for broader structure rather than local structure

    # Plots
    sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts'], save=f'{section}_afterqc.png')
    sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=["n_genes_by_counts", "total_counts"], save=f'{section}_afterqc_spatial.png')
    featureplot(adata, markers, save=f'{section}.png', use_raw=False) # False to allow scaled values to be plotted from adata.X
    plt.close()

    # Leiden clustering at multiple resolutions and differential gene expression
    for resolution in resolutions:
        str_res = str(resolution).replace('.', '_')
        leiden_key = 'leiden' + str_res
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
        sc.pl.pca(adata, color=leiden_key, save=f'{section}_{str_res}_pca_plot.png')
        plot_umap(adata, resolution)
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=leiden_key, save=f'{section}_{str_res}_clustered_spatial.png')
        plt.close()
        sc_degs(adata, resolution, plots=['dotplot', 'matrixplot', 'stacked_violin'])

    # Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
    cluster_stats(adata, resolutions, scores = ['Calinski-Harabasz', 'Davies-Bouldin'])

    return adata


# In[ ]:


def remove_area(adata, coords: tuple, color: str='celltype'):
    '''
    Permanently removes a given area of the image
    coords specified just like crop_coord in sq.pl.spatial_scatter()
    (xmin, ymin, xmax, ymax)
    '''
    xmin, ymin, xmax, ymax = coords
    adata = adata[~((adata.obs['cell_x'] > xmin) & (adata.obs['cell_x'] < xmax) & (adata.obs['cell_y'] > ymin) & (adata.obs['cell_y'] < ymax))].copy()
    sq.pl.spatial_scatter(adata, library_id='spatial', shape=None, color=color)
    return adata


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


def plot_overlay_panels(
    adata,
    section: str,
    tiff_path,
    npz_path,
    cluster_key="leiden",
    mask_type="nuclei_exp",   # options: "nuclei" or "nuclei_exp"
    win_coords=None,          # (xmin, ymin, xmax, ymax), overrides win_size if provided
    win_size=2500,
    output_path=None 
):
    
    """
    Plot 4-panel figure for a given G4X sample:
      (1) Cropped H&E with zoom box
      (2) Zoomed H&E
      (3) Zoomed H&E + segmentation mask colored by clusters
      (4) Spatial scatter with zoom box

    Dependencies
    -----------
    import numpy as np
    import scanpy as sc
    import squidpy as sq
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from tifffile import imread
    from pathlib import Path

    Parameters
    ----------
    adata : AnnData
        AnnData object of a single section
    section : str
        Section identifier.
    tiff_path : str
        path to HE.ome.tiff file
    npz_path : str
        path to segmentation_mask.npz file
    cluster_key : str
        Column name in adata.obs giving cluster assignments (e.g. "leiden").
    mask_type : str
        Which segmentation mask to use: "nuclei" (nuclear mask) or "nuclei_exp" (expanded nuclei/cell boundaries).
    win_coords : tuple, optional
        (xmin, ymin, xmax, ymax) in full-image coordinates.
    win_size : int
        Window size if win_coords not provided.
    """
    # load H&E image
    print("Loading H&E image")
    img = imread(tiff_path)

    # load segmentation mask
    print("Loading segmentation mask")
    mask_npz = np.load(npz_path)
    if mask_type not in mask_npz.files:
        raise ValueError(f"Mask type {mask_type} not found. Available mask types are: {mask_npz.files}")
    mask = mask_npz[mask_type]
    print(f"Using segmentation mask: {mask_type}")

    # crop H&E to spatial bounding box
    coords = adata.obsm["spatial"]
    xmin_sp, ymin_sp = coords.min(axis=0).astype(int)
    xmax_sp, ymax_sp = coords.max(axis=0).astype(int)

    he_cropped = img[ymin_sp:ymax_sp, xmin_sp:xmax_sp, :]
    print("Original H&E shape:", img.shape)
    print("Cropped H&E shape:", he_cropped.shape)

    # define zoom window
    if win_coords is None:
        y0, x0 = mask.shape[0] // 2, mask.shape[1] // 2
        ymin, ymax = y0 - win_size // 2, y0 + win_size // 2
        xmin, xmax = x0 - win_size // 2, x0 + win_size // 2
    else:
        xmin, ymin, xmax, ymax = win_coords

    # segmentation window
    mask_win = mask[ymin:ymax, xmin:xmax]

    # H&E window (shift into cropped coords)
    he_win = he_cropped[(ymin - ymin_sp):(ymax - ymin_sp),
                        (xmin - xmin_sp):(xmax - xmin_sp), :]

    # map segmentation IDs to chosen cluster labels
    id_to_cluster = {}
    for _, row in adata.obs.iterrows():
        try:
            seg_id = int(row["cell_id"].split("-")[1])
            id_to_cluster[seg_id] = row[cluster_key]
        except Exception:
            continue

    cluster_win = np.full_like(mask_win, fill_value=-1, dtype=int)
    nonzero = mask_win > 0
    cluster_win[nonzero] = [id_to_cluster.get(x, -1) for x in mask_win[nonzero]]

    # assign colors
    clusters_present = np.unique(cluster_win[cluster_win >= 0])
    clusters_present = sorted(clusters_present, key=lambda x: int(x))

    palette = sc.pl.palettes.default_20
    colors = {str(c): palette[int(c) % len(palette)] for c in clusters_present}

    overlay_rgb = np.zeros((*cluster_win.shape, 4))
    for c in clusters_present:
        mask_c = cluster_win == int(c)
        color_rgba = plt.matplotlib.colors.to_rgba(colors[str(c)], alpha=0.75)
        overlay_rgb[mask_c] = color_rgba

    # ----------------------------
    # Plot 4 panels
    # ----------------------------
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # (1) Cropped H&E with zoom box
    axes[0].imshow(he_cropped)
    rect_full = mpl.patches.Rectangle(
        (xmin - xmin_sp, ymin - ymin_sp),
        xmax - xmin,
        ymax - ymin,
        linewidth=3,          # thicker outline
        edgecolor="black",    # black box
        facecolor="none"
    )
    axes[0].add_patch(rect_full)
    axes[0].set_title(f"{sample_id} - Cropped H&E with zoom")
    axes[0].axis("off")

    # (2) Zoomed H&E
    axes[1].imshow(he_win)
    axes[1].set_title("H&E (zoomed)")
    axes[1].axis("off")

    # (3) Zoomed H&E + clusters
    axes[2].imshow(he_win)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"H&E (zoomed) + {cluster_key}")
    axes[2].axis("off")

    # (4) Full spatial scatter
    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        shape=None,
        color=[cluster_key],
        ax=axes[3]
    )
    rect = mpl.patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=3,          # thicker outline
        edgecolor="black",    # black box
        facecolor="none"
    )
    axes[3].add_patch(rect)
    axes[3].set_title(f"Spatial scatter with zoom ({cluster_key})")

    plt.tight_layout()

    # ---- save (treat output_path as a stem; only strip .png/.pdf if present)
    if output_path is not None:
        out = Path(output_path)
        if out.suffix.lower() in {".png", ".pdf"}:
            out = out.with_suffix("")  # remove only .png/.pdf if user passed one
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out.as_posix() + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(out.as_posix() + ".pdf", bbox_inches="tight")
        print(f"Saved → {out}.png and {out}.pdf")

    plt.show()


# In[ ]:


def cluster_section(adata, markers: dict, resolutions: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], use_rep='X_pca'):
    
    adata = adata.copy()
    
    # get section information
    section = adata.uns['section']
    print(section)
    
    sc.pp.scale(adata, max_value=10) # scale to unit variance and zero mean and clip any gene expression above 10 std deviations

    # Feature Plots
    featureplot(adata, markers, save=f'{section}.png', use_raw=False) # False to allow scaled values to be plotted from adata.X
    plt.close()

    # Leiden clustering at multiple resolutions and differential gene expression
    for resolution in resolutions:
        str_res = str(resolution).replace('.', '_')
        leiden_key = 'leiden' + str_res
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=resolution, key_added=leiden_key)
        plot_umap(adata, resolution)
        sq.pl.spatial_scatter(adata, library_id="spatial", shape=None, color=leiden_key, save=f'{section}_{str_res}_clustered_spatial.png')
        plt.close()
        sc_degs(adata, resolution, plots=['dotplot', 'matrixplot', 'stacked_violin'], use_rep=use_rep)

    # Calculate clustering statistics to see which is the optimal resolution from a mathematical perspective
    cluster_stats(adata, resolutions, scores = ['Calinski-Harabasz', 'Davies-Bouldin'], rep=use_rep)

    return adata


# In[ ]:


def plot_g4x_inset(sdata, section: str, obs_key: str, palette: dict, xmin: int, ymin: int, width: int, height: int):
    '''
    sdata: spatialdata object to plot. Must have annotated adata stored in adata.tables['table']
    section: name of the section that sdata represents. Only used in the filename.
    obs_key: name of the adata.obs key used for coloring the segmentation mask
    palette: dictionary of colors with order corresponding to the order of the ordered subtypes in obs_key
    xmin: minimum x coordinate
    ymin: minimum y coordinate
    width: width of ROI
    height: height of the ROI
    
    Makes 5 plots:
    1) segmentation mask with red rectangle marking ROI
    2) H&E with red rectangle marking ROI
    3) magnified H&E
    4) magnified segmentation
    5) magnified segmentation overlaid on magnified H&E

    2025-11-5 breaking change: made palette a dict, not a list, to keep colors in the correct order
    '''
    
    adata = sdata.tables['table']
    groups = list(palette.keys()) # these are the unique cell types in that section
    palette_colors = list(palette.values())
    
    # define the cropping coordinates
    bb_xmin = xmin
    bb_ymin = ymin
    bb_w = width
    bb_h = height
    bb_xmax = bb_xmin + bb_w
    bb_ymax = bb_ymin + bb_h

    # 1) plot the segmentation mask with red rectangle outlining the ROI
    f, ax = plt.subplots(figsize=(10, 10))
    sdata.pl.render_shapes(color=obs_key, groups=groups, palette=palette_colors).pl.show(ax=ax)
    rect = mpl.patches.Rectangle((bb_xmin, bb_ymin), bb_w, bb_h, linewidth=5, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.axis('off') # remove frame and axis markings
    ax.set_title('') # remove title
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., frameon=False)
    plt.savefig(f'{section}_{obs_key}_with_rect.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{section}_{obs_key}_with_rect.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # 2) plot the H&E with red rectangle outlining the ROI
    f, ax = plt.subplots(figsize=(10, 10))
    sdata.pl.render_images('hne').pl.show(ax=ax, frameon=False)
    rect = mpl.patches.Rectangle((bb_xmin, bb_ymin), bb_w, bb_h, linewidth=5, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.axis('off') # remove frame and axis markings
    ax.set_title('') # remove title
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., frameon=False)
    plt.savefig(f'{section}_{obs_key}_hne_with_rect.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{section}_{obs_key}_hne_with_rect.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a cropped sdata in the ROI
    cropped_sdata = sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[bb_xmin, bb_ymin],
        max_coordinate=[bb_xmax, bb_ymax],
        target_coordinate_system="global",
    )

    # 3) Plot the segmentation of the ROI
    fig, ax = plt.subplots(figsize=(10, 10))
    cropped_sdata.pl.render_shapes('cell_boundaries', color=obs_key, groups=groups, palette=palette_colors).pl.show(ax=ax)
    ax.axis('off') # remove frame and axis markings
    ax.set_title('') # remove title
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., frameon=False)
    plt.savefig(f'{section}_{obs_key}_inset.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{section}_{obs_key}_inset.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # 4) Plot the H&E of the ROI
    fig, ax = plt.subplots(figsize=(10, 10))
    cropped_sdata.pl.render_images("hne").pl.show(ax=ax)
    ax.axis('off') # remove frame and axis markings
    ax.set_title('') # remove title
    plt.savefig(f'{section}_{obs_key}_hne_inset.png', dpi=300)
    plt.savefig(f'{section}_{obs_key}_hne_inset.pdf', dpi=300)
    plt.close()

    # 5) Plot the H&E and the segmentation of the ROI together
    fig, ax = plt.subplots(figsize=(10, 10))
    cropped_sdata.pl.render_images("hne").pl.show(ax=ax)
    cropped_sdata.pl.render_shapes('cell_boundaries', color=obs_key, groups=groups, palette=palette_colors, fill_alpha=0.7).pl.show(ax=ax)
    ax.axis('off') # remove frame and axis markings
    ax.set_title('') # remove title
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., frameon=False)
    plt.savefig(f'{section}_{obs_key}_inset_with_hne.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{section}_{obs_key}_inset_with_hne.pdf', dpi=300, bbox_inches='tight')
    plt.close()


# In[ ]:




