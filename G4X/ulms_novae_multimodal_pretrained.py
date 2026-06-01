#!/usr/bin/env python
# coding: utf-8

# # Novae for the G4X dataset
# - Using CONCH for H&E integration
# - Fine-tune a pretrained model

# # Set up

# ## Dependencies

# In[ ]:


import os
from pathlib import Path
import scanpy as sc
import spatialdata as sd
import spatialdata_plot # You need this to plot spatialdata objects. Otherwise you get an Attribute Error
from tifffile import imread
import novae
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import torch

print("scanpy:", sc.__version__)
print("spatialdata:", sd.__version__)
print("novae:", novae.__version__)


# In[ ]:


novae.settings.scale_to_microns = 0.3125
torch.set_float32_matmul_precision('high')
torch.cuda.is_available()


# In[ ]:


# Token to access the CONCH and TITAN models on HuggingFace
from huggingface_hub import login
login(token="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", add_to_git_credential=True)


# # Parameters

# In[ ]:


N_DOMAINS = 8
MAX_EPOCHS = 20


# ## Functions

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


# Function reads in all the h5ad files from directory to separate anndata objects
def import_data(directory): 
    # List all files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('.h5ad')]
    data_list = []

    for file in files:
        # Load the AnnData object
        path = os.path.join(directory, file)
        adata = sc.read_h5ad(path)
        adata.uns['filename'] = str(file).strip('.h5ad')
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


def connect_adata_to_sdata(adata, sdata, include_points=True):
    '''
    Adds or replaces the table of a spatialdata object with a new adata
    sdata: the existing spatialdata object
    adata: the new anndata that you want to put into your spatialdata object instead of the one that is already there

    Connects the new anndata to the segmentation mask and H&E image
    2025-11-25 scale points as well
    2026-03-16 include_points argument since it will bring the transcripts array into memory
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
    if include_points:
        sd.transformations.set_transformation(sdata.points["transcripts"], scale, to_coordinate_system="global")
    
    return sdata


# ## Directories

# In[ ]:


CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
OUTPUT_MASTER_DIR = create_output_dir(PARENT_DIR, 'novae_multimodal_pretrained', change_dir=True)

ANNDATA_DIR = PARENT_DIR / 'ad'

DATA_DIR = PARENT_DIR.parent.parent / 'G4X/G4X_raw'
print(f"DATA_DIR is {DATA_DIR}")

# This is where spatialdata objects will be stored
SD_OUTPUTS = create_output_dir(PARENT_DIR, 'sd')


# # Load the data

# In[ ]:


adatas = import_data(ANNDATA_DIR)
adatas.sort(key=lambda x: x.uns['filename'])
for adata in adatas:
    print(adata.uns['filename'])


# In[ ]:


# Reformattting and cleaning up to save memory
cols_to_drop = ['_indices', '_scvi_batch', '_scvi_ind_x', '_scvi_labels', 
                'leiden0_1', 'leiden0_2', 'leiden0_3', 'leiden0_4', 'leiden0_5', 
                'leiden0_6', 'leiden0_7', 'leiden0_8', 'leiden0_9', 'leiden1_0', 
                'leiden1_1', 'leiden1_2', 'leiden1_3', 'leiden1_4', 'leiden1_5']

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

sdata_fns = [] # store list of file paths to spatial data objects
for adata in adatas:
    section = adata.uns['filename']
    print(section)
    adata.obs.drop(cols_to_drop, axis='columns', inplace=True, errors='ignore')
    for key in uns_to_drop:
        if key in adata.uns:
            adata.uns.pop(key, None)
    
    del adata.obsm['distance_neighbor']
    del adata.obsm['index_neighbor']
    del adata.obsp['connectivities']
    del adata.obsp['distances']
    del adata.layers['X_normalized_resolVI']

    # Novae needs counts or log1p and a 'spatial' obsm key
    adata.X = adata.layers['counts']
    adata.obsm['spatial'] = adata.obsm.pop('X_spatial') # need this if anndata came out of resolVI
    print(adata)
    
    # Create the spatialdata object
    sdata = load_g4x_zarr(DATA_DIR, section, proseg_folder_label='proseg_no_igj')
    sdata = connect_adata_to_sdata(adata, sdata, include_points=True)
    print(sdata)

    # Write the spatialdata object
    sdata_fn = SD_OUTPUTS / f"{section}.zarr"
    sdata.write(sdata_fn, overwrite=True)
    sdata_fns.append(sdata_fn)


# In[ ]:


# clean up for memory management
del adatas
del adata
del sdata
del sdata_fn


# In[ ]:


sdatas = []
for sdata_fn in sdata_fns:
    # Reload the spatialdata object
    sdata = sd.read_zarr(sdata_fn)
    print(sdata)
    adata = sdata["table"]
    section = adata.uns['filename']
    
    # check that coordinate system is lined up properly
    f, ax = plt.subplots(figsize=(10, 10))
    sdata.pl.render_images().pl.render_shapes(color='coarse_celltype').pl.show(ax=ax)
    ax.axis('off') # remove frame and axis markings
    ax.set_title('') # remove title
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., frameon=False)
    plt.savefig(f'{section}_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compute the patch embeddings using a computer vision model (CONCH). One embedding per patch.
    # Each embedding is an H&E representation of the of the cell neighborhood. Multiple cells may share the same embedding.
    # Run this for each sdata individually
    print(f"Computing histo embeddings for section {section}")
    novae.compute_histo_embeddings(sdata, model="conch", device="cuda")    
    sdatas.append(sdata)


# In[ ]:


# Compute PCA of these embeddings and assign each cell to the embedding of the closest patch
# Run this on a list of sdatas
novae.compute_histo_pca(sdatas)


# In[ ]:


del sdata
adatas = []
for sdata in sdatas:
    adata = sdata["table"]
    section = adata.uns['filename']
    print(section)
    
    # Plot the CONCH embeddings (patch embeddings, not embeddings per cell)
    try: 
        # dimension number 0 of the CONCH embeddings
        dim = 0
        sc.pl.spatial(sdata["conch_embeddings"], color=str(dim), spot_size=10, save=f'{section}_CONCH_embeddings_dim{dim}.png')

        # dimension number 0 of the embeddings projected over the cells (in Xenium micron coordinate system)
        adata.obs[f"pca_dim{dim}_cells"] = adata.obsm["histo_embeddings"][:, dim]
        sc.pl.spatial(adata, color=f"pca_dim{dim}_cells", spot_size=10, save=f"{section}_pca_dim{dim}_cells.png")
    except Exception as e:
        print(e)

    adatas.append(adata)


# In[ ]:


del adata
for adata in adatas:
    slide_name = adata.uns['filename']
    print(adata.obsm) # should have key histo_embeddings
    novae.spatial_neighbors(adata, radius=80, slide_key='Section')
    print(f"Finished computing spatial neighbors for {slide_name}")
    
    novae.plot.connectivities(adata, show=False)
    plt.savefig(f'{slide_name}_neighbors.png')
    plt.close()


# In[ ]:


del adata
model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
print(model)


# In[ ]:


# Fine tune the model
model.fine_tune(adatas, max_epochs=MAX_EPOCHS, accelerator='cuda', num_workers=4, logger=CSVLogger("logs"), log_every_n_steps=10)
print(model)

# save the logs in a directory called "logs"
novae.plot.loss_curve("logs")
plt.savefig("novae_logs.png", dpi=300, bbox_inches='tight')


# In[ ]:


model.compute_representations(adatas, accelerator="cuda", num_workers=4)


# In[ ]:


model.assign_domains(adatas, level=N_DOMAINS)

print(f"Plotting domain proportions")
novae.plot.domains_proportions(adatas, show=False) # can add obs_key if needed
plt.savefig(f'{N_DOMAINS}_proportions.png')
plt.close()


# In[ ]:


# Get a batch-corrected embedding
model.batch_effect_correction(adatas)


# In[ ]:


output_dir = create_output_dir(OUTPUT_MASTER_DIR, "novae_ad_multimodal") # directory for saving novae-processed anndatas
hallmark = str(PARENT_DIR / 'ref/h.all.v2026.1.Hs.json') # hallmark json file
reactome = str(PARENT_DIR / 'ref/c2.cp.reactome.v2026.1.Hs.json') # reactome json file

# Loop through the anndatas
for adata in adatas:
    slide_name = adata.uns['filename']

    try:
        print(f"Plotting domains for {slide_name}")
        novae.plot.domains(adata, slide_name_key="Section", cell_size=20, show=False)
        plt.savefig(f'{slide_name}_{N_DOMAINS}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)

    try:
        # PAGA graphs (trajectory inference) of the domains on each section
        print(f"Plotting PAGA graph for {slide_name}")
        novae.plot.paga(adata, show=False)
        plt.savefig(f'{slide_name}_{N_DOMAINS}_paga.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)

    try:
        # SVGs for each section
        print(f"Plotting SVGs for {slide_name}")
        novae.plot.spatially_variable_genes(adata, top_k=8, vmax="p95", cell_size=20, show=False)
        plt.savefig(f'{slide_name}_{N_DOMAINS}_svgs.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)

    try:
        # spatial pathway analysis for each domain per section - Hallmark
        print(f"Plotting Hallmark pathway scores for {slide_name}")
        novae.plot.pathway_scores(adata, pathways=hallmark, figsize=(10, 12), show=False)
        plt.savefig(f'{slide_name}_{N_DOMAINS}_hallmark.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)

    try:
        # spatial pathway analysis for each domain per section - Reactome
        print(f"Plotting Reactome pathway scores for {slide_name}")
        novae.plot.pathway_scores(adata, pathways=reactome, figsize=(10, 12), show=False)
        plt.savefig(f'{slide_name}_{N_DOMAINS}_reactome.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)

    # Save
    adata_path = output_dir / f'{slide_name}_ulms_novae.h5ad'
    adata.write_h5ad(adata_path)
    print(f"Wrote anndata for {slide_name} to {adata_path}")


# In[ ]:


model.save_pretrained("ulms_novae_model_multimodal_pretrained")
print("Saved fine-tuned ULMS Novae model")

