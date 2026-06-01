# Slingshot trajectory inference on the ULMS tumor subset
# https://bioconductor.org/packages/release/bioc/vignettes/slingshot/inst/doc/vignette.html
# https://github.com/YosefLab/Thymus_CITE-seq/blob/main/Pseudotime/Slingshot_thymus_posselecting_filtered.Rmd
# https://nbisweden.github.io/workshop-archive/workshop-scRNAseq/2020-01-27/labs/compiled/slingshot/slingshot.html
# https://bustools.github.io/BUS_notebooks_R/slingshot.html
# Run this on the scVI UMAP directly, not on the scVI embedding, for computational efficiency

library(dplyr)
library(ggplot2)
library(Seurat, lib.loc = "/scg/apps/software/r/4.3.3/lib") # make sure it's Seurat v5
library(slingshot, lib.loc = "/scg/apps/software/r/4.3.3/lib")
library(RColorBrewer)

# set up directories
setwd("..") # set working directory to the parent directory
getwd() # check working directory
proj_dir <- getwd()
output_dir <-  "slingshot_umap"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}
setwd(output_dir)
getwd() # set working directory to the newly created directory

# load previously created tumor subset R object with cluster labels and scVI UMAP
ulms <- readRDS(file.path(proj_dir, 'objects', 'for_seurat', 'tumor_subset.Rds'))
print(ulms)

# extract the scVI embedding
rd <- Embeddings(ulms[["umap"]])
print(dim(rd))

# extract cluster labels
cl <- factor(ulms@meta.data$leiden0_2, levels = as.character(0:8))
print(length(cl))

# plot umap with the correct colors. Make sure they are in cluster order.
leiden0_2_colors <- c(
  "#4c72b0",  # ESR1+ cells
  "#c44e52",  # IFN signaling cells
  "#dd8452",  # Mesenchyme-like cells
  "#937860",  # Cycling cells
  "#55a868",  # AR+/PGR+ cells
  "#8172b3",  # Ischemic cells
  "#da8bc3",  # SMC-like cells
  "#8c8c8c",  # Neuron-like cells
  "#ccb974"  # ATRX+/DAXX- cells
)
DimPlot(ulms, group.by = 'leiden0_2', cols=leiden0_2_colors) %>% ggsave("umap_leiden0_2.png", .)
names(leiden0_2_colors) = as.character(0:8)

###############################################################################
# run slingshot without a set starting point
set.seed(1234)

lin <- getLineages(rd, cl)
print(lin)
# Detailed check
cat("Start cluster(s):", slingParams(lin)$start.clus, "\n")
cat("End cluster(s):", slingParams(lin)$end.clus, "\n")
cat("\nLineages:\n")
print(slingLineages(lin))
cat("\nMST edges:\n")
print(slingMST(lin, as.df = TRUE))

# Plot the lineages if you ran slingshot directly on the UMAP
png("lineage_no_start.png")
plot(rd, col=leiden0_2_colors, cex=0.1, asp=0.8, pch=16)
lines(SlingshotDataSet(lin), lwd=3, col='black')
dev.off()

# Get the pseudotime curves
crv <- getCurves(lin, approx_points=150)
print(crv)

# Extract pseudotime and visualize on UMAP
# slingPseudotime() returns an n by L matrix representing each cell's pseudotime along each lineage.
pt <- slingPseudotime(crv)
for (i in seq_len(ncol(pt))) { # print all the lineages
  col_name <- paste0("pseudotime_lineage_", i)
  ulms[[col_name]] <- pt[, i]
  
  p <- FeaturePlot(ulms, features = col_name) +
    scale_color_viridis_c(na.value = "grey85") +
    ggtitle(paste("Lineage", i))
  ggsave(paste0("no_start_pseudotime_lineage_", i, "_umap.png"), p)
}

# Plot the pseudotime curves if you ran slingshot directly on the UMAP
png("curve_no_start.png")
plot(rd, col=leiden0_2_colors, cex=0.1, asp=0.8, pch=16)
lines(SlingshotDataSet(crv), lwd=3, col='black')
dev.off()

###############################################################################
# run slingshot with a set starting point
# starting point cluster 0
set.seed(1234)

lin <- getLineages(rd, cl, start.clus='0')
print(lin)
# Detailed check
cat("Start cluster(s):", slingParams(lin)$start.clus, "\n")
cat("End cluster(s):", slingParams(lin)$end.clus, "\n")
cat("\nLineages:\n")
print(slingLineages(lin))
cat("\nMST edges:\n")
print(slingMST(lin, as.df = TRUE))

# Plot the lineages if you ran slingshot directly on the UMAP
png("lineage_start_0.png")
plot(rd, col=leiden0_2_colors, cex=0.1, asp=0.8, pch=16)
lines(SlingshotDataSet(lin), lwd=3, col='black')
dev.off()

# Get the pseudotime curves
# Prevent curve from extending beyond the center of the start cluster
# https://github.com/kstreet13/slingshot/issues/68
crv <- getCurves(lin, approx_points=150, extend = 'n')
print(crv)

# Extract pseudotime and visualize on UMAP
# slingPseudotime() returns an n by L matrix representing each cell's pseudotime along each lineage.
pt <- slingPseudotime(crv)
for (i in seq_len(ncol(pt))) { # print all the lineages
  col_name <- paste0("pseudotime_lineage_", i)
  ulms[[col_name]] <- pt[, i]
  
  p <- FeaturePlot(ulms, features = col_name) +
    scale_color_viridis_c(na.value = "grey85") +
    ggtitle(paste("Lineage", i))
  ggsave(paste0("start0_pseudotime_lineage_", i, "_umap.png"), p)
}

# Plot the pseudotime curves if you ran slingshot directly on the UMAP
png("curve_start_0.png")
plot(rd, col=leiden0_2_colors, cex=0.1, asp=0.8, pch=16)
lines(SlingshotDataSet(crv), lwd=3, col='black')
dev.off()

sessionInfo()