#!/usr/bin/env python
# coding: utf-8

# # Novae for the G4X dataset
# - This one just uses the transcriptomics, not the H&Es
# - Changed to list of anndatas because I did not have success with one big anndata

# # Set up

# ## Dependencies

# In[ ]:


import os
from pathlib import Path
import scanpy as sc
import novae
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import torch

print("scanpy:", sc.__version__)
print("novae:", novae.__version__)


# In[ ]:


novae.settings.scale_to_microns = 0.3125
torch.set_float32_matmul_precision('high')
torch.cuda.is_available()


# # Parameters

# In[ ]:


N_DOMAINS = 5
MAX_EPOCHS = 20
NUM_WORKERS = 4


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


# ## Directories

# In[ ]:


CURRENT_DIR = Path.cwd()
PARENT_DIR = CURRENT_DIR.parent
OUTPUT_MASTER_DIR = create_output_dir(PARENT_DIR, 'novae', change_dir=True)

ANNDATA_DIR = PARENT_DIR / 'ad'

DATA_DIR = PARENT_DIR.parent.parent / 'G4X/G4X_raw'
print(f"DATA_DIR is {DATA_DIR}")


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

for adata in adatas:
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
    
    print(adata.uns['filename'])
    print(adata)


# # Compute spatial neighbors

# In[ ]:


for adata in adatas:
    slide_name = adata.uns['filename']
    novae.spatial_neighbors(adata, radius=80, slide_key='Section')
    print(f"Finished computing spatial neighbors for {slide_name}")

    novae.plot.connectivities(adata, show=False)
    plt.savefig(OUTPUT_MASTER_DIR / f'{slide_name}_neighbors.png')
    plt.close()


# # Load and fine-tune the model

# In[ ]:


model = novae.Novae.from_pretrained("MICS-Lab/novae-human-0")
model


# In[ ]:


print("Fine tuning the model for better performance")
# Fine tune the model
model.fine_tune(adatas, max_epochs=MAX_EPOCHS, accelerator='cuda', num_workers=4, logger=CSVLogger("logs"), log_every_n_steps=1)
print(model)

# save the logs in a directory called "logs"
novae.plot.loss_curve("logs")
plt.savefig(OUTPUT_MASTER_DIR / "novae_logs.png", dpi=300, bbox_inches='tight')
plt.close()


# In[ ]:


model.compute_representations(adatas, accelerator='cuda', num_workers=NUM_WORKERS)


# # Domains and downstream analysis

# In[ ]:


model.assign_domains(adatas, level=N_DOMAINS) # if you fine-tuned make sure to use the level argument

print(f"Plotting domain proportions")
novae.plot.domains_proportions(adatas, show=False) # can add obs_key if needed
plt.savefig(f'{N_DOMAINS}_proportions.png', bbox_inches='tight')
plt.close()


# In[ ]:


# Get a batch-corrected embedding
model.batch_effect_correction(adatas)


# In[ ]:


output_dir = create_output_dir(OUTPUT_MASTER_DIR, "novae_ad") # directory for saving novae-processed anndatas
hallmark = str(PARENT_DIR / 'ref/h.all.v2026.1.Hs.json') # hallmark json file
reactome = str(PARENT_DIR / 'ref/c2.cp.reactome.v2026.1.Hs.json') # reactome json file

# Loop through the anndatas
for adata in adatas:
    slide_name = adata.uns['filename']

    try:
        print(f"Plotting domains for {slide_name}")
        novae.plot.domains(adata, slide_name_key="Section", cell_size=15, show=False)
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
        novae.plot.spatially_variable_genes(adata, top_k=8, vmax="p95", cell_size=15, show=False)
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
    adata_path = output_dir / f'{slide_name}_{N_DOMAINS}_ulms_novae.h5ad'
    adata.write_h5ad(adata_path)
    print(f"Wrote anndata for {slide_name} to {adata_path}")


# In[ ]:


model.save_pretrained("ulms_novae_model")
print("Saved fine-tuned ULMS Novae model")

