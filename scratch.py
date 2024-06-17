import os, sys
import polars as pl
import matplotlib.pylab as plt, seaborn as sns

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk, DE
from utils import Timer, print_df, savefig, debug
from ryp import r, to_r

debug(third_party=True)
data_dir = 'projects/def-wainberg/single-cell/Maze/SLCKO'
working_dir = 'projects/def-wainberg/karbabi/maze-slcko'

# Cell type annotation with reference atlas ####################################
# Load the SLCKO human kidney single cell 
with Timer ('[SLCKO] Loading and QCing single cell data'):
    sc_query = SingleCell(f'{data_dir}/slcko_10x.h5ad', num_threads=None)\
        .with_columns_obs(batch = pl.lit('Maze24'))\
        .qc(cell_type_confidence_column=None, 
            doublet_column='is_doublet',
            allow_float=True)
    sc_query_raw = sc_query 

# Load the Novella-Rausell 2023 integrated atlas of healthy mouse kidney
# doi.org/10.1016/j.isci.2023.106877
with Timer ('[Novella-Rausell] Loading and QCing single cell data'):
    sc_ref = SingleCell(
        f'{data_dir}/mouse_kidney_atlas.h5ad', 
        num_threads=None)\
        .set_var_names('feature_name')\
        .cast_var({'feature_name': pl.String})\
        .cast_obs({'Origin': pl.String, 'author_cell_type': pl.String})\
        .with_columns_obs(
            batch = pl.col.Origin, 
            cell_type_fine = pl.col.author_cell_type,
            cell_type_broad = pl.when(
                pl.col.author_cell_type.is_in([
                    'PTS1', 'PTS2', 'PTS3', 'PTS3T2'])).then(pl.lit('PT'))
                .otherwise(pl.col.author_cell_type))\
        .qc(cell_type_confidence_column=None,
            doublet_column=None, 
            max_mito_fraction=0.2,
            custom_filter=pl.col.cell_type_fine.ne('Unknown') & 
                pl.col.cell_type_broad.ne('Unknown'),
            allow_float=True)

# Find highly variable genes, normalize expression, then run PCA
with Timer('Highly variable genes'):
    sc_query, sc_ref = sc_query.hvg(
        sc_ref, batch_column='batch', allow_float=True)

with Timer('Normalize'):
    sc_query = sc_query.normalize(allow_float=True, num_threads=None)
    sc_ref = sc_ref.normalize(allow_float=True, num_threads=None)
    
with Timer('PCA'):
    sc_query, sc_ref = sc_query.PCA(sc_ref, verbose=True)

# Plot PC1 vs PC2
with Timer('Plot PCA'):
    plt.scatter(
        sc_query.obsm['PCs'][:, 0], sc_query.obsm['PCs'][:, 1],
        label='Neptune', s=1, alpha=0.05, rasterized=True)
    plt.scatter(
        sc_ref.obsm['PCs'][:, 0], sc_ref.obsm['PCs'][:, 1],
        label='Lake Atlas', s=1, alpha=0.05, rasterized=True)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    savefig(f'{working_dir}/figures/compare_pcs.png')

# Harmonize the principal components between the two datasets with Harmony:
# github.com/slowkow/harmonypy
# Note: the order of the two datasets doesn't matter   
with Timer('Harmony'):
    sc_query, sc_ref = sc_query.harmonize(
        sc_ref, batch_column='batch', pytorch=True, num_threads=None)

# Generate new PCA plots after harmonization
with Timer('Plot PCA'):
    plt.scatter(
        sc_query.obsm['Harmony_PCs'][:, 0], 
        sc_query.obsm['Harmony_PCs'][:, 1],
        label='Neptune', s=1, alpha=0.05, rasterized=True)
    plt.scatter(
        sc_ref.obsm['Harmony_PCs'][:, 0],
        sc_ref.obsm['Harmony_PCs'][:, 1],
        label='Lake Atlas', s=1, alpha=0.05, rasterized=True)
    plt.xlabel('Harmony PC1')
    plt.ylabel('Harmony PC2')
    savefig(f'{working_dir}/figures/compare_pcs_harmony.png')
    
# Transfer cell-type labels from Novella-Rausell et al. to SLCKO
with Timer('Label transfer'):
    sc_query = sc_query\
        .label_transfer_from(
            sc_ref, cell_type_column='cell_type_fine',
            cell_type_confidence_column='cell_type_fine_confidence',
            num_neighbors=10)\
        .label_transfer_from(
            sc_ref, cell_type_column='cell_type_broad',
            cell_type_confidence_column='cell_type_broad_confidence',
            num_neighbors=10)
    sc_query = sc_query.with_columns_obs(
        pl.col.passed_QC & 
        pl.col.cell_type_fine.len().over(
            pl.when('passed_QC').then('cell_type_fine')).ge(300) &
        pl.col.cell_type_broad.len().over(
            pl.when('passed_QC').then('cell_type_broad')).ge(300))

# Generate and plots UMAP
with Timer('UMAP plots'):
    sc_query = sc_query\
        .with_columns_obs(passed_QC_fine=pl.col.passed_QC &
                pl.col.cell_type_fine_confidence.ge(0.8))\
        .UMAP(QC_column='passed_QC_fine',
              UMAP_key='UMAP_fine', num_threads=24)
    sc_query.plot_UMAP(
        'cell_type_fine', 
        f'{working_dir}/figures/sc_query_cell_type_fine_umap.svg',
        cells_to_plot_column='passed_QC_fine', UMAP_key='UMAP_fine',
        label=True, label_kwargs={'size': 6},
        legend=True, legend_kwargs={'fontsize': 'x-small', 'ncols': 1})
    sc_query = sc_query\
        .with_columns_obs(
            passed_QC_broad=pl.col.passed_QC &
                pl.col.cell_type_broad_confidence.ge(0.8))\
        .UMAP(QC_column='passed_QC_broad',
              UMAP_key='UMAP_broad', num_threads=24)
    sc_query.plot_UMAP(
        'cell_type_broad', 
        f'{working_dir}/figures/sc_query_cell_type_broad_umap.png',
        cells_to_plot_column='passed_QC_broad', UMAP_key='UMAP_broad',
        label=True, label_kwargs={'size': 6},
        legend=True, legend_kwargs={'fontsize': 'x-small', 'ncols': 1})
    # sc_query.plot_UMAP(
    #     'cell_type_fine_confidence',
    #     f'{working_dir}/figures/sc_query_cell_type_confidence_umap.png',
    #     UMAP_key='UMAP_fine')
    # sc_query.plot_UMAP(
    #     'sample_id', f'{working_dir}/figures/sc_query_sample_umap.png',
    #     UMAP_key='UMAP_fine', legend=False)

# Save labelled single cell data 
with Timer('Saving single cell'):
    sc_query.X = sc_query_raw.X
    sc_query.save(f'{data_dir}/slcko_10x_labelled.h5ad', overwrite=True)
    
    sc_query_filt = sc_query.filter_obs(
        pl.col.passed_QC_fine & pl.col.passed_QC_broad)\
        .with_columns_obs(cs.categorical().cast(pl.String))\
        .drop_obs(['batch', 'orig.ident', 'notes'])
    sc_query_filt.obsm['X_umap'] = sc_query_filt.obsm['UMAP_broad']
    sc_query_filt.save(f'{data_dir}/slcko_10x_cellxgene.h5ad', 
                  preserve_strings=True, overwrite=True) 

# Peudobulk differential expression testing ####################################
# Pseudobulk the data 

with Timer('[SLCKO] Pseuduobulking'):
    sc_query = SingleCell(f'{data_dir}/slcko_10x_labelled.h5ad')\
        .cast_obs({'sample_id': pl.String, 'condition': pl.String,
                   'cell_type_fine': pl.String, 'cell_type_broad': pl.String})
    pb = sc_query.pseudobulk(
            ID_column='sample_id', cell_type_column='cell_type_fine',
            QC_column='passed_QC_fine') | \
        sc_query.filter_obs(pl.col.cell_type_broad.eq('PT'))\
            .pseudobulk(
            ID_column='sample_id', cell_type_column='cell_type_broad',
            QC_column='passed_QC_broad')
    pb = pb.with_columns_obs(
            WT_AAI_vs_WT=pl.when(pl.col.condition.eq('WT')).then(0)
                .when(pl.col.condition.eq('WT_AAI')).then(1)
                .otherwise(None),
            SLCKO_AAI_vs_SLCKO=pl.when(pl.col.condition.eq('SLCKO')).then(0)
                .when(pl.col.condition.eq('SLCKO_AAI')).then(1)
                .otherwise(None),
            WT_DAPA_vs_WT=pl.when(pl.col.condition.eq('WT')).then(0)
                .when(pl.col.condition.eq('WT_DAPA')).then(1)
                .otherwise(None),
            SLCKO_vs_WT=pl.when(pl.col.condition.eq('WT')).then(0)
                .when(pl.col.condition.eq('SLCKO')).then(1)
                .otherwise(None),
            SLCKO_AAI_vs_WT_AAI=pl.when(pl.col.condition.eq('WT_AAI')).then(0)
                .when(pl.col.condition.eq('SLCKO_AAI')).then(1)
                .otherwise(None),
            WT_AAI_DAPA_vs_WT_AAI=pl.when(pl.col.condition.eq('WT_AAI')).then(0)
                .when(pl.col.condition.eq('WT_AAI_DAPA')).then(1)
                .otherwise(None))\
        .drop_cell_types(['ATL', 'DTL', 'MC', 'B lymph', 'Asc-Vasa-Recta', 
                          'Per', 'PEC', 'LOH'])    
                
contrasts = ['WT_AAI_vs_WT', 'SLCKO_AAI_vs_SLCKO', 'WT_DAPA_vs_WT', 
             'SLCKO_vs_WT', 'SLCKO_AAI_vs_WT_AAI', 'WT_AAI_DAPA_vs_WT_AAI']

# Differential expression testing 
for contrast in contrasts:
    with Timer(f'[Neptune] Differential expression for {contrast}'):
        pb_filt = pb\
            .qc(case_control_column=contrast, 
                custom_filter=pl.col(contrast).is_not_null(),
                verbose=False)
        drop_cell_types = []
        drop_cell_types.extend([
            cell_type for cell_type, (_, obs, _) in pb_filt.items() 
            if any(count < 3 for count in 
                   obs[contrast].value_counts()['count'])])
        pb_filt = pb_filt.drop_cell_types(drop_cell_types)
        de = pb_filt\
            .DE(label_column=contrast, 
                case_control=True,
                covariate_columns=None,
                include_library_size_as_covariate=True,
                include_num_cells_as_covariate=True,
                verbose=False)
        de.plot_voom(f'{working_dir}/figures/voom/{contrast}', 
                     overwrite=True, PNG=True)
        de.save(f'{working_dir}/output/DE_{contrast}', overwrite=True)   
        de.table.write_csv(f'{working_dir}/output/DE_{contrast}.csv')
        print_df(de.get_num_hits(threshold=0.1).sort('num_hits'))
    
with Timer('[SLCKO] Sample-level PCA'):
    lcpm = pb.qc(case_control_column=None, verbose=False).log_CPM()
    r('plot_list = list()') 
    for cell_type, (X, obs, var) in lcpm.items():
        to_r(cell_type, 'cell_type'); to_r(X, 'X'); to_r(obs, 'obs')
        r('''
            library(ggplot2)
            library(ggsci)
            
            mat = X[, order(-apply(X, 2, var))[1:2000]]
            pca_result = prcomp(mat, center = T, scale = F, rank=2) 
            pca_df = as.data.frame(pca_result$x)
            pca_df$sample_id = obs$sample_id
            pca_df$condition = obs$condition 

            plot_list[[cell_type]] =
                ggplot(pca_df, 
                    aes(x = PC1, y = PC2, color = condition,
                        fill = condition)) +
                    geom_point(size = 3) +
                    stat_ellipse(geom = "polygon", level=0.25, 
                        alpha = 0.3) + 
                    labs(x = "PCA 1", y = "PCA 2", color = "Condition",
                        title=cell_type) + 
                    scale_color_frontiers() + 
                    scale_fill_frontiers(guide = "none") + 
                    theme_classic() +
                    theme(axis.text = element_text(size = 12, 
                            color = "black"),
                        axis.title = element_text(size = 16),
                        plot.title = element_text(size = 20, face = "bold", 
                            hjust = 0.5))''')
    to_r(working_dir, 'working_dir')
    r('''
      png(paste0(working_dir, "/figures/sample_level_pca.png"),
        height = 15, width = 25, units = "in", res = 400)
      ggpubr::ggarrange(plotlist = plot_list, 
        common.legend = TRUE, legend = "right")
      dev.off()
      ''')
    
    
'''
 cell_type        cell_type_confidence 
 DTL              0.432577             
 ATL              0.484858             
 MC               0.493569             
 NK               0.551152             
 DC               0.552247             
 Vas-Efferens     0.566038             
 Per              0.587597             
 LOH              0.592974             
 B lymph          0.599776             
 MD               0.608458             
 Neutro           0.616667             
 CD-Trans         0.620513             
 Vas-Afferens     0.630909             
 PTS3             0.659735             
 Desc-Vasa-Recta  0.66087              
 PEC              0.664436             
 PTS2             0.668953             
 DCT-CNT          0.674527             
 CTAL             0.703255             
 PTS1             0.706296             
 DTL-ATL          0.708082             
 Asc-Vasa-Recta   0.71374              
 MTAL             0.740265             
 Glom-Endo        0.740886             
 Macro            0.762437             
 Fib              0.771688             
 CNT              0.777185             
 DCT              0.782693             
 T lymph          0.811607             
 PTS3T2           0.851005             
 PC               0.876571             
 Endo             0.878345             
 ICA              0.923316             
 ICB              0.929377             
 Podo             0.984956             
 null             NaN  
'''


sc_query.obs.group



print_df(sc_query.obs.group_by('condition')
        .agg(pl.col('cell_type_fine_confidence').mean())
        .sort('cell_type_fine_confidence'))














sc_query.obs = sc_query.obs.drop(['cell_type', 'cell_type_confidence',
                                  'passed_QC'])
sc_query = sc_query.qc(cell_type_confidence_column=None, 
            doublet_column='is_doublet',
            allow_float=True)
del sc_query.obsm['UMAP']

print_df(sc_query.filter_obs(pl.col.passed_QC).obs['cell_type_fine']\
    .value_counts().sort('count'))


print_df(pl.DataFrame({
    "Column Name": sc_ref.var.columns,
    "Data Type": [str(dtype) for dtype in sc_ref.var.dtypes]
}))


print_df(sc_query.obs.group_by(['cell_type_fine', 'condition']).count()\
    .sort('cell_type_fine'))