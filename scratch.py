import os, sys
import polars as pl, polars.selectors as cs
import pandas as pd, numpy as np
import matplotlib.pylab as plt, seaborn as sns

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk, DE
from utils import Timer, print_df, savefig, get_coding_genes, debug
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
            cell_type = pl.col.author_cell_type)\
        .qc(cell_type_confidence_column=None,
            doublet_column=None, 
            max_mito_fraction=0.15,
            custom_filter=pl.col.cell_type.ne('Unknown'),
            allow_float=True)

# Find highly variable genes, normalize expression, then run PCA
with Timer('Highly variable genes'):
    sc_query, sc_ref = sc_query.hvg(
        sc_ref, batch_column='batch', allow_float=True)

with Timer('Normalize'):
    sc_query = sc_query.normalize(allow_float=True)
    sc_ref = sc_ref.normalize(allow_float=True)
    
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
        sc_ref, cell_type_column='cell_type',
        cell_type_confidence_column='cell_type_confidence',
        min_cell_type_confidence=0.80,
        num_neighbors=10,
        num_index_neighbors=60)
    sc_query = sc_query.with_columns_obs(
        pl.col.passed_QC & 
        pl.col.cell_type.len().over(
            pl.when('passed_QC').then('cell_type')).ge(50))
    print_df(sc_query.obs.group_by('cell_type')
          .agg(pl.col('cell_type_confidence').mean())
          .sort('cell_type_confidence'))
    sns.ecdfplot(data=sc_query.obs, x='cell_type_confidence')
    savefig(f'{working_dir}/figures/sc_query_cell_type_confidence_ecdf.png')

# Generate and plots UMAP
with Timer('UMAP plots'):
    sc_query = sc_query.UMAP(num_threads=24)
    sc_query.plot_UMAP(
        'cell_type', 
        f'{working_dir}/figures/sc_query_cell_type_umap.png',
        label=True, label_kwargs={'size': 6},
        legend=True, legend_kwargs={'fontsize': 'x-small', 'ncols': 1})
    sc_query.plot_UMAP(
        'cell_type_confidence',
        f'{working_dir}/figures/sc_query_cell_type_confidence_umap.png')
    sc_query.plot_UMAP(
        'sample_id', f'{working_dir}/figures/sc_query_sample_umap.png',
        legend=False)
    sc_ref = sc_ref.UMAP(num_threads=24)
    sc_ref.plot_UMAP(
        'cell_type', 
        f'{working_dir}/figures/sc_ref_cell_type_umap.png',
        label=True, label_kwargs={'size': 6},
        legend=True, legend_kwargs={'fontsize': 'x-small', 'ncols': 2})
    
# Save labelled single cell data 
with Timer('Saving single cell'):
    sc_query.X = sc_query_raw.X
    sc_query.obsm['X_umap'] = sc_query.obsm['UMAP'] # for CellXGene
    sc_query.save(f'{data_dir}/slcko_10x_labelled.h5ad', overwrite=True)

# Peudobulk differential expression testing ####################################
# Pseudobulk the data 

with Timer('[SLCKO] Pseuduobulking'):
    # sc_query = SingleCell(f'{data_dir}/slcko_10x_labelled.h5ad')
    pb = sc_query\
        .cast_obs({'sample_id': pl.String, 'condition': pl.String,
                   'cell_type': pl.String})\
        .pseudobulk(ID_column='sample_id', cell_type_column='cell_type')\
        .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
        .with_columns_obs(
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
                .otherwise(None))
        
with Timer('[SLCKO] Pseuduobulking'):

        

label_column = [
    'Additive_Genotype', 'Dominant_Genotype', 'Recessive_Genotype']
case_control_column = {
    'Additive_Genotype': None, 'Dominant_Genotype': 'Dominant_Genotype', 
    'Recessive_Genotype': 'Recessive_Genotype'}
case_control = {
    'Additive_Genotype': False, 'Dominant_Genotype': True, 
    'Recessive_Genotype': True}

for label in label_column:
    with Timer(f'[Neptune] Differential expression for {label}'):
        de = pb\
            .qc(case_control_column=case_control_column[label], 
                custom_filter=pl.col(label).is_not_null())\
            .DE(label_column=label, 
                case_control=case_control[label],
                covariate_columns=['eGFR_Bx', 'Age', 'Sex'],
                include_library_size_as_covariate=True,
                include_num_cells_as_covariate=True,
                verbose=False)
        de.plot_voom(f'{working_dir}/figures/voom/{label}', 
                     overwrite=True, PNG=True)
        de.save(f'{working_dir}/output/DE_{label}', overwrite=True)   
        de.table.write_csv(f'{working_dir}/output/DE_{label}/table.csv')
        print(label)
        print_df(de.get_num_hits(threshold=0.1).sort('cell_type'))






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





















sc_query.obs = sc_query.obs.drop(['cell_type', 'cell_type_confidence',
                                  'passed_QC'])
sc_query = sc_query.qc(cell_type_confidence_column=None, 
            doublet_column='is_doublet',
            allow_float=True)
del sc_query.obsm['UMAP']

print_df(sc_query.filter_obs(pl.col.passed_QC).obs['cell_type']\
    .value_counts().sort('count'))


print_df(pl.DataFrame({
    "Column Name": sc_ref.var.columns,
    "Data Type": [str(dtype) for dtype in sc_ref.var.dtypes]
}))
