import os, scvi, warnings, pickle
import scanpy as sc, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scvi.model.utils import mde
warnings.filterwarnings("ignore")

os.chdir('projects/def-wainberg/karbabi/maze-slcko')
data_dir = '../../single-cell/Maze/SLCKO'

ad_ref = sc.read_h5ad(f'{data_dir}/mouse_kidney_atlas.h5ad')
ad_ref.var.index = ad_ref.var['feature_name']
ad_ref.obs = pd.DataFrame({
    'cell_type': ad_ref.obs['author_cell_type'],
    'batch': ad_ref.obs['Origin'],
    'pct_counts_mt': ad_ref.obs['pct_counts_mt']})

ad_query = sc.read_h5ad(f'{data_dir}/10x_sc_all.h5ad')
ad_query.obs = pd.DataFrame({
    'cell_type': 'Unknown', 'batch': 'Maze24',
    'pct_counts_mt': ad_query.obs['percent_mt']})

ad = sc.concat((ad_query, ad_ref))
sc.pp.highly_variable_genes(
    ad, flavor='seurat_v3', n_top_genes=2000, 
    batch_key='batch', subset=True)

ad_query = ad[ad.obs.batch == 'Maze24'].copy()
ad_ref = ad[~(ad.obs.batch == 'Maze24')].copy()

scvi.model.SCVI.setup_anndata(
    ad_ref, batch_key='batch', 
    labels_key='cell_type',
    continuous_covariate_keys=['pct_counts_mt'])

vae = scvi.model.SCVI(ad_ref, n_layers=2)
vae.train()

# with open('output/vae.pkl', 'wb') as file:
#     pickle.dump((vae), file)
with open('output/vae.pkl', 'rb') as file:
    vae = pickle.load(file)
    
scanvae = scvi.model.SCANVI.from_scvi_model(
    vae, unlabeled_category = "Unknown")
print("Labelled Indices: ", len(scanvae._labeled_indices))
print("Unlabelled Indices: ", len(scanvae._unlabeled_indices))
    
scanvae.train(max_epochs=20)

# with open('output/scanvae.pkl', 'wb') as file:
#     pickle.dump((scanvae), file)
with open('output/scanvae.pkl', 'rb') as file:
    scanvae = pickle.load(file)
    
model = scvi.model.SCANVI.load_query_data(ad_query, scanvae)
model._unlabeled_indices = np.arange(ad_query.n_obs)
model._labeled_indices = []
print("Labelled Indices: ", len(model._labeled_indices))
print("Unlabelled Indices: ", len(model._unlabeled_indices))

model.train(
    max_epochs=20,
    plan_kwargs=dict(weight_decay=0.0),
    check_val_every_n_epoch=10)
model.save('output/model')

query_latent = sc.AnnData(model.get_latent_representation())
query_latent.obs['cell_type'] = model.predict().tolist()
query_latent.obs['transfer_score'] = model.predict(soft = True)\
    .max(axis = 1).tolist()
sc.pp.neighbors(query_latent)
sc.tl.umap(query_latent)

sc.set_figure_params(dpi_save=400, vector_friendly=False, figsize=[9,8])
sc.pl.umap(query_latent,
           color=['cell_type'],
           legend_loc = 'on data',
           legend_fontoutline=1.2,
           frameon=False,
           ncols=1, size=2,
           save='figures/model_umap_cell_type.png')

sns.kdeplot(data = query_latent.obs['transfer_score'].dropna(), 
            cumulative = True)
plt.savefig('figures/transfer_score_cdf.png')

plt.ecdf(query_latent.obs['transfer_score'].dropna())
plt.savefig('figures/transfer_score_cdf.png')


sc.pl.umap(query_latent,
        color=['transfer_score'],
        frameon=False,
        ncols=1,
        save='figures/model_umap_transfer_score.png')















latent = sc.AnnData(vae.get_latent_representation())
latent.obs["cell_type"] = ad.obs['cell_type'].tolist()
latent.obs["batch"] = ad.obs['batch'].tolist()
sc.pp.neighbors(latent, n_neighbors=8)
sc.tl.umap(latent)
sc.pl.umap(latent,
           color=['batch', 'cell_type'],
           frameon=False,
           ncols=1,
           save='figures/scvi_umap.png')

lvae = scvi.model.SCANVI.from_scvi_model(
    vae, adata = ad, 
    labels_key = 'cell_type',
    unlabeled_category = 'Unknown')

lvae.train(max_epochs=20, n_samples_per_label=100)

with open('output/lvae.pkl', 'wb') as file:
    pickle.dump((lvae), file)
with open('output/lvae.pkl', 'rb') as file:
    lvae = pickle.load(file)

ad.obs['predicted'] = lvae.predict(ad)
ad.obs['transfer_score'] = lvae.predict(soft = True).max(axis = 1)
ad = ad[ad.obs.Batch == 'Maze24']







vae = scvi.model.SCVI.load(
    f'{data_dir}/scVI_model_full', f'{data_dir}/mouse_kidney_atlas_hvg_filt.h5ad')
vae.view_anndata_setup()


ad_query = sc.read_h5ad(f'{data_dir}/10x_sc_all.h5ad')
ad_query.obs['Celltype_finest_lowres'] = 'Unknown'
ad_query.obs['Origin'] = 'Unknown'
ad_query.obs['Source'] = 'Cell'
ad_query.obs['pct_counts_mt'] = ad_query.obs['percent_mt']

ad_query._validate_anndata()
adata_manager = vae.get_anndata_manager(ad_query, required=True)

scvi.model.SCVI.prepare_query_anndata(ad_query, vae)
vae = scvi.model.SCVI.load_query_data(ad_query, vae)









ad_ref = sc.read_h5ad(f'{data_dir}/mouse_kidney_atlas_hvg_filt.h5ad')
ref_model = scvi.model.SCANVI.load(
    f'{data_dir}/scANVI_model_full', adata=ad_ref)
ref_model.view_anndata_setup()

ref_model.setup_anndata(ad_ref, 
    batch_key='Origin', 
    labels_key='Celltype_finest_lowres', 
    unlabeled_category='Unknown', 
    categorical_covariate_keys=None,
    continuous_covariate_keys=['pct_counts_mt'])

ad_query = sc.read_h5ad(f'{data_dir}/10x_sc_all.h5ad')
ad_query.obs['Celltype_finest_lowres'] = 'Unknown'
ad_query.obs['Origin'] = 'Unknown'
ad_query.obs['Source'] = 'Cell'
ad_query.obs['pct_counts_mt'] = ad_query.obs['percent_mt']

scvi.model.SCANVI.prepare_query_anndata(ad_query, ref_model)
ad_query = scvi.model.SCANVI.load_query_data(
    ad_query, ref_model, freeze_dropout=True)


ref_latent = sc.AnnData(ref_model.get_latent_representation())
ref_latent.obs["cell_type"] = ad_ref.obs['Celltype_finest_lowres'].tolist()
ref_latent.obs["batch"] = ad_ref.obs['Origin'].tolist()

sc.pp.neighbors(ref_latent, n_neighbors=8)
sc.tl.leiden(ref_latent)
sc.tl.umap(ref_latent)
sc.pl.umap(ref_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,)

ref_model.setup_anndata()

len(ref_model._labeled_indices)
len(ref_model._unlabeled_indices)

ad_query_orig = ad_query.copy()
ad_query.obs = pd.DataFrame({
    'Celltype_finest_lowres': 'Unknown',
    'Origin': 'Unknown',
    'Source': 'Cell',
    'pct_counts_mt': ad_query.obs['percent_mt']
}, index=ad_query.obs.index)



ad.obsm['X_scVI'] = vae.get_latent_representation()
ad.obsm['X_scVI_mde'] = mde(ad.obsm['X_scVI'])
sc.pl.embedding(
    ad,
    basis='X_scVI_mde',
    color=['batch'],
    frameon=False,
    ncols=1,
)