# # scANVI to integrate ULMS G4X dataset into ULMS scRNAseq reference
# - decided to run this on all the cells
# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/scarches_scvi_tools.html

# This version uses multi-GPU training for the scANVI reference mapping step, which is the slowest step.

# SET UP DEPENDENCIES

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


# Required so that multi-GPU trainig doesn't run the whole script twice
def main():
    print(f"\nRunning script: {Path(__file__).name}\n")

    print("Is CUDA available?", torch.cuda.is_available())

    # version control
    print("\nPackage versions:")
    print("torch:", torch.__version__)
    print("anndata:", ad.__version__)
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
    print("scanpy:", sc.__version__)
    print("scvi:", scvi.__version__)


    mpl.rcParams['pdf.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
    mpl.rcParams['ps.fonttype'] = 42 # TrueType font for editing in Adobe Illustrator
    plt.rcParams['axes.facecolor'] = 'white'
    plt.ioff()
    sc.settings.autoshow = False
    sc.settings.n_jobs = -1  # Use all available cores
    SEED = 1234
    scvi.settings.seed = SEED
    torch.set_float32_matmul_precision("high")

    # SET UP DIRECTORIES

    CURRENT_DIR = Path.cwd()
    PARENT_DIR = CURRENT_DIR.parent
    print(PARENT_DIR)

    G4X_DIR = PARENT_DIR.parent / 'G4X'
    print(G4X_DIR)

    SCRNASEQ_DIR = PARENT_DIR.parent / 'scRNAseq'
    print(SCRNASEQ_DIR)

    # Making an output directory using the pathlib package
    OUTPUT_MASTER_DIR = jpascvi.create_output_dir(PARENT_DIR, 'scANVI_all_multigpu', change_figdir=True)

    # USER-DEFINED PARAMETERS

    early_stopping_kwargs = {
        'check_val_every_n_epoch': 1,
        'early_stopping': True,
        'early_stopping_patience': 20, # how many epochs of no change are tolerated
        'early_stopping_monitor': "elbo_validation"
    }

    multi_gpu_kwargs = {
        'accelerator': "gpu",
        'devices': -1,
        'strategy': "ddp_find_unused_parameters_true",
        'check_val_every_n_epoch': 1,
    }

    # The conditions key specify the covariates over which to integrate your samples
    # batch is the sequencing batch (scRNAseq) 
    # section (G4X, like in resolVI)
    # sample is the consistent patient sample numbering (G4X and scRNAseq)
    # assay is scRNAseq or spatial
    tech_key = 'assay'
    condition_keys = ['batch', 'sample', 'section']
    cell_type_key = 'cell_type'
    SCRNASEQ_MIN_COUNTS = 10 # remove scRNAseq cells with fewer than this many counts after subsetting to the G4X gene list, which will be the genes used for integration
    SCVI_MAX_EPOCHS = 100
    SCANVI_MAX_EPOCHS = 100
    QUERY_MAX_EPOCHS = 200
    SCVI_LATENT_KEY = "X_scVI"
    SCANVI_LATENT_KEY = "X_scANVI"
    SCANVI_PREDICTIONS_KEY = "scanvi_pred"
    # https://github.com/scverse/scvi-tools/issues/2726
    QUERY_TRAIN_BATCH_SIZE = 1024

    # LOAD THE G4X QUERY DATA

    # # Import G4X data, which will eventually be the query data
    print("\nLoading G4X anndata")
    data_dir = G4X_DIR / 'objects'
    print(data_dir)
    g4x_adata = sc.read_h5ad(data_dir / 'g4x_raw_counts.h5ad')
    print(g4x_adata)

    # reformat the G4X anndata
    print("\nReformatting G4X anndata")
    g4x_adata.obs_names = g4x_adata.obs['cell_name']
    g4x_adata.obs[tech_key] = 'spatial'
    g4x_adata.obs.rename(columns={'Sample' : 'sample', 'Section' : 'section'}, inplace=True)
    g4x_adata.obs['cell_type'] = 'Unknown'
    g4x_adata.obs['batch'] = 'g4x'
    g4x_adata.obs['batch'] = g4x_adata.obs['batch'].astype('category')
    gene_list = g4x_adata.var_names.tolist()
    adata_query = g4x_adata
    del g4x_adata
    print(adata_query)

    # add in the celltype annotations
    print("\nLoading G4X annotations")
    data_dir = G4X_DIR / 'annotation'
    print(data_dir)
    g4x_ann = sc.read_h5ad(data_dir / 'scviva_celltype.h5ad')
    g4x_ann.obs_names = g4x_ann.obs['cell_name']
    print(g4x_ann)
    adata_query.obs['cell_type'] = g4x_ann.obs.loc[adata_query.obs.index, 'celltype']
    print(adata_query.obs['cell_type'])
    print(np.unique(adata_query.obs['cell_type']))
    adata_query = adata_query[adata_query.obs['cell_type'] != 'Necrosis'].copy()
    adata_query.obs['cell_type'] = adata_query.obs['cell_type'].cat.remove_unused_categories()
    adata_query.obs['cell_type'] = adata_query.obs['cell_type'].cat.rename_categories(
        {'Macrophage': 'Myeloid'}
    )
    print(np.unique(adata_query.obs['cell_type']))
    print(adata_query)
    del g4x_ann

    # LOAD THE SCRNASEQ REFERENCE DATA

    # # Import scRNAseq data, which will be the reference data
    print("\nLoading scRNAseq anndata")
    data_dir = SCRNASEQ_DIR / 'objects'
    print(data_dir)
    adata = ad.read_h5ad(data_dir / 'annotated_raw_counts.h5ad')
    print(adata)

    # Reformat the scRNAseq anndata for training, subsetting to only those genes present in the G4X anndata
    print("\nReformatting scRNAseq anndata")
    adata.obs[tech_key] = 'scRNAseq'
    adata.obs.rename(columns={'celltype' : 'cell_type'}, inplace=True)
    adata = adata[adata.obs['cell_type'] != 'RBC'].copy()
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    adata.obs['cell_type'] = adata.obs['cell_type'].cat.remove_unused_categories()
    adata.obs['sample'] = adata.obs['sample'].str.replace('Sample', '')
    adata.obs['sample'] = adata.obs['sample'].astype('category')
    adata.obs['section'] = 'scRNAseq'
    adata.obs['section'] = adata.obs['section'].astype('category')
    # subset to the G4X gene list
    ref_genes = set(adata.var_names)
    # gene_list is the G4X gene list from above. Now filter for only those genes that are also in the scRNAseq reference, which will be the genes used for integration
    gene_list = [gene for gene in gene_list if gene in ref_genes]
    adata = adata[:, gene_list].copy()
    print(adata)
    adata_ref = adata
    del adata
    print(adata_ref)

    # Remove cells with low counts after subsetting to the G4X gene list, which will be the genes used for integration
    print("\nFiltering scRNAseq cells with low counts after subsetting to G4X gene list")
    sc.pp.filter_cells(adata_ref, min_counts=SCRNASEQ_MIN_COUNTS)
    print(adata_ref)

    # Note: make sure the cell_type categories are named the same thing in reference and query
    # e.g. not T_cells and T_cell
    print(adata_ref.obs['cell_type'].cat.categories)
    # Intersection
    print(set(adata_query.obs['cell_type'].cat.categories) & set(adata_ref.obs['cell_type'].cat.categories))
    # Union
    print(set(adata_query.obs['cell_type'].cat.categories) | set(adata_ref.obs['cell_type'].cat.categories))

    # in case there are any genes in the G4X that are not in the scRNAseq, though that is unlikely
    assert all(g in adata_query.var_names for g in adata_ref.var_names), "Gene mismatch!"
    print(adata_query)
    adata_query = adata_query[:, adata_ref.var_names.tolist()].copy()
    print(adata_query)

    # Prepare for scVI/scANVI
    print("\nFinal scRNAseq anndata:")
    print(adata_ref)
    print("\nFinal G4X anndata:")
    print(adata_query)

    # Log normalize data and save raw counts in a layer, as recommended for scvi-tools
    adata_ref.layers["counts"] = adata_ref.X.copy() # this layer will contain the raw counts
    sc.pp.normalize_total(adata_ref) # normalize X to the median total counts
    sc.pp.log1p(adata_ref) # logarithmize X
    adata_ref.raw = adata_ref # full dimension normalized logtransformed raw data

    # SCVI TRAINING OF THE REFERENCE DATA
    print("\nTraining scVI reference model\n")
    scvi_ref_dir = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scvi_ref', change_figdir=True)

    # # Train the reference scVI model on fully labeled scRNAseq data
    # batch_key=tech_key is a single value here but necessary for scArches 
    # to handle the new 'spatial' batch during query mapping
    scvi.model.SCVI.setup_anndata(adata_ref, 
                                layer="counts", 
                                batch_key=tech_key, 
                                categorical_covariate_keys=condition_keys)
    # Using custom parameters found to work well in scArches
    scvi_ref = scvi.model.SCVI(adata_ref, 
                            gene_likelihood="nb", # choosing this since spatial will be less sparse
                            use_layer_norm="both", # scArches found that using layer norm in both the encoder and decoder worked best for integration
                            use_batch_norm="none", # scArches found that not using batch norm in the decoder worked best for integration
                            encode_covariates=True, # Necesary for scArches to map new batches in the encoder
                            dropout_rate=0.2,
                            n_layers=2,)
    print(scvi_ref)

    # Train the vae with early stopping for the default number of epochs
    # Use only 1 GPU for training the reference model since it is relatively small and multi-GPU training can be unstable for small datasets
    scvi.settings.seed = 1234
    scvi_ref.train(max_epochs=SCVI_MAX_EPOCHS, 
                accelerator="gpu",
                devices=1,
                **early_stopping_kwargs)
    try:
        save_path = scvi_ref_dir / 'scvi_ref_elbo_plot.png'
        jpascvi.check_training(scvi_ref, save=save_path)
    except Exception as e:
        print(f"Error occurred while checking training: {e}")
        print(f"type(scvi_ref.history): {type(scvi_ref.history)}")

    # Get the latent representation of the reference data
    adata_ref.obsm[SCVI_LATENT_KEY] = scvi_ref.get_latent_representation()
    sc.pp.neighbors(adata_ref, use_rep=SCVI_LATENT_KEY)
    sc.tl.umap(adata_ref, min_dist=0.3, random_state=SEED)

    # Visual check of integration
    sc.pl.umap(
        adata_ref,
        color=cell_type_key,
        show=False,
        frameon=False,
        save=f'{cell_type_key}.png'
    )
    sc.pl.umap(
        adata_ref,
        color='sample',
        show=False,
        frameon=False,
        save='sample.png'
    )
    sc.pl.umap(
        adata_ref,
        color='batch',
        show=False,
        frameon=False,
        save='batch.png'
    )
    # save the model and anndata with the latent representation
    scvi_ref.save(scvi_ref_dir, prefix="scvi_ref_", save_anndata=True, overwrite=True)
    print("scVI reference model saved to ", scvi_ref_dir)


    # TRAINING THE SCANVI REFERENCE MODEL
    print("\nTraining scANVI reference model\n")
    scanvi_ref_dir = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_ref', change_figdir=True)

    scanvi_ref = scvi.model.SCANVI.from_scvi_model(
        scvi_ref,
        adata=adata_ref,
        labels_key=cell_type_key,
        unlabeled_category="Unknown"
    )
    scvi.settings.seed = 1234
    scanvi_ref.train(max_epochs=SCANVI_MAX_EPOCHS, 
                    accelerator="gpu",
                    devices=1,
                    **early_stopping_kwargs)
    try: 
        save_path = scanvi_ref_dir / 'scanvi_ref_elbo_plot.png'
        jpascvi.check_training(scanvi_ref, save=save_path)
    except Exception as e:
        print(f"Error occurred while checking training: {e}")
        print(f"type(scanvi_ref.history): {type(scanvi_ref.history)}")

    # Get the latent representation of the reference data
    adata_ref.obsm[SCANVI_LATENT_KEY] = scanvi_ref.get_latent_representation()
    sc.pp.neighbors(adata_ref, use_rep=SCANVI_LATENT_KEY)
    sc.tl.umap(adata_ref, min_dist=0.3, random_state=SEED)

    # Visual check of integration
    sc.pl.umap(
        adata_ref,
        color=cell_type_key,
        show=False,
        frameon=False,
        save=f'{cell_type_key}.png'
    )
    sc.pl.umap(
        adata_ref,
        color='sample',
        show=False,
        frameon=False,
        save='sample.png'
    )
    sc.pl.umap(
        adata_ref,
        color='batch',
        show=False,
        frameon=False,
        save='batch.png'
    )
    # save the model and anndata with the latent representation
    scanvi_ref.save(scanvi_ref_dir, prefix="scanvi_ref_", save_anndata=True, overwrite=True)
    print("scANVI reference model saved to ", scanvi_ref_dir)


    # # REFERENCE MAPPING OF QUERY G4X DATA ONTO AN INTEGRATED SCRNASEQ REFERENCE ATLAS USING SCANVI
    print("\nMapping query G4X data onto reference atlas with scANVI\n")
    scanvi_query_dir = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_query', change_figdir=True)

    # Log normalize data and save raw counts in a layer, as recommended for scvi-tools
    adata_query.layers["counts"] = adata_query.X.copy() # this layer will contain the raw counts
    sc.pp.normalize_total(adata_query) # normalize X to the median total counts
    sc.pp.log1p(adata_query) # logarithmize X
    adata_query.raw = adata_query # full dimension normalized logtransformed raw data

    # Prepare the query anndata by reordering the genes and padding any missing genes with zeros
    scvi.model.SCANVI.prepare_query_anndata(adata_query, scanvi_ref)
    # Online update of reference model using the scArches algorithm
    scanvi_query = scvi.model.SCANVI.load_query_data(adata_query, scanvi_ref)

    # Train the model
    # Weight decay of 0.0 ensures the latent representation of the reference cells remains the same
    # Try multi-GPU training
    # Per documentation early stopping is not supported for multi-GPU training, but we can set it to check validation every epoch and then check training manually after
    # https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/use_cases/multiGPU.html
    # https://docs.scvi-tools.org/en/1.3.3/user_guide/use_case/multi_gpu_training.html
    scvi.settings.seed = 1234
    scanvi_query.train(max_epochs=QUERY_MAX_EPOCHS, 
                    plan_kwargs={"weight_decay": 0.0},
                    **multi_gpu_kwargs,
                    batch_size=QUERY_TRAIN_BATCH_SIZE,
                    )
    try:
        save_path = scanvi_query_dir / 'scanvi_query_elbo_plot.png'
        jpascvi.check_training(scanvi_query, save=save_path)
    except Exception as e:
        print("Error in checking training:", e)
        print(f"type(scanvi_query.history): {type(scanvi_query.history)}")

    # Get the latent space
    adata_query.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation()
    sc.pp.neighbors(adata_query, use_rep=SCANVI_LATENT_KEY)
    sc.tl.umap(adata_query, min_dist=0.3, random_state=SEED)

    # Visual check of integration
    sc.pl.umap(
        adata_query,
        color=cell_type_key,
        show=False,
        frameon=False,
        save=f'{cell_type_key}.png'
    )
    sc.pl.umap(
        adata_query,
        color='sample',
        show=False,
        frameon=False,
        save='sample.png'
    )
    sc.pl.umap(
        adata_query,
        color='section',
        show=False,
        frameon=False,
        save='section.png'
    )
    # save the model and anndata with the latent representation
    scanvi_query.save(scanvi_query_dir, prefix="scanvi_query_", save_anndata=True, overwrite=True)
    print("scANVI query model saved to ", scanvi_query_dir)

    # SCANVI CELL TYPE PREDICTIONS
    print("\nGetting scANVI cell type predictions for query G4X data\n")
    # Get the cell type predictions for the query data
    adata_query.obs[SCANVI_PREDICTIONS_KEY] = scanvi_query.predict()

    # Check the cell type predictions
    try: 
        df = adata_query.obs.groupby([cell_type_key, SCANVI_PREDICTIONS_KEY]).size().unstack(fill_value=0)
        norm_df = df / df.sum(axis=0)

        plt.figure(figsize=(14, 12))
        plt.pcolor(norm_df)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xlabel("Predicted")
        plt.ylabel("Observed")
        save_path = scanvi_query_dir / 'cell_type_prediction_confusion_matrix.png'
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print("Error in plotting cell type prediction confusion matrix:", e)


    # ANALYZING THE QUERY AND REFERENCE TOGETHER
    print("\nAnalyzing query and reference together\n")
    SCANVI_FULL_DIR = jpascvi.create_output_dir(OUTPUT_MASTER_DIR, 'scanvi_full', change_figdir=True)

    adata = ad.concat([adata_query, adata_ref], join='inner')
    print(adata)

    full_predictions = scanvi_query.predict(adata)
    print(f"Acc: {np.mean(full_predictions == adata.obs[cell_type_key])}")

    adata.obs[SCANVI_PREDICTIONS_KEY] = full_predictions

    sc.pp.neighbors(adata, use_rep=SCANVI_LATENT_KEY)
    sc.tl.umap(adata, min_dist=0.3, random_state=SEED)

    sc.pl.umap(
        adata,
        color=tech_key,
        show=False,
        frameon=False,
        save=f'{tech_key}.png'
    )
    sc.pl.umap(
        adata,
        color='sample',
        show=False,
        frameon=False,
        save='sample.png'
    )
    sc.pl.umap(
        adata,
        color='batch',
        show=False,
        frameon=False,
        save='batch.png'
    )
    sc.pl.umap(
        adata,
        color='section',
        show=False,
        frameon=False,
        save='section.png'
    )
    sc.pl.umap(
        adata,
        color=cell_type_key,
        show=False,
        frameon=False,
        save=f'{cell_type_key}.png'
    )
    sc.pl.umap(
        adata,
        color=SCANVI_PREDICTIONS_KEY,
        show=False,
        frameon=False,
        save=f'{SCANVI_PREDICTIONS_KEY}.png'
    )

    adata_path = SCANVI_FULL_DIR / 'scanvi_full_adata.h5ad'
    adata.write_h5ad(adata_path)
    print("Full anndata with query and reference together saved to ", adata_path)


if __name__ == "__main__":
    main()