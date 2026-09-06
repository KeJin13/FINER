

import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler


from .expression_reconstruction_utils import (
    get_R_matrix, process_image_features, build_typewise_blockdiag_laplacian_sparse_sym,
    compute_env_features, seed_everything, Structure_Model, to_torch, torch_sparse_to_scipy,
    solve_harmonic_extension, optimize_residuals_post_hoc,
    _to_dense
)




def parse_args():
    parser = argparse.ArgumentParser(description="Expression Reconstruction")
    
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--tissue_name", type=str, default="tissue_name", help="Name of the tissue sample folder")
    parser.add_argument("--save_name", type=str, default="finer_reconstructed.h5ad", help="Name of the output h5ad file")
    
    parser.add_argument("--st_filename", type=str, default="ST_adata.h5ad", help="ST h5ad filename inside tissue_name folder")
    parser.add_argument("--scrna_filename", type=str, default="sc_adata_log.h5ad", help="scRNA-seq h5ad filename inside tissue_name folder")
    
    parser.add_argument("--spot_radius", type=float, default=129.4, help="Spot radius")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lambda_morph", type=float, default=1.0, help="Lambda morphology parameter")
    parser.add_argument("--lambda_gamma", type=float, default=0.1, help="Lambda gamma parameter")
    parser.add_argument("--seed", type=int, default=32, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    args.sample_dir = os.path.join(args.data_dir, args.tissue_name)
    
    args.cell_h5ad = os.path.join(args.sample_dir, "cellular_scaffold.h5ad")
    args.feat_hipt = os.path.join(args.sample_dir, "hipt_sub_feats.npy")
    args.feat_resccl = os.path.join(args.sample_dir, "retccl_feats.npy")
    
    args.st_h5ad = os.path.join(args.sample_dir, args.st_filename)
    args.scrna_h5ad = os.path.join(args.sample_dir, args.scrna_filename)
        
    return args



def main():
    args = parse_args()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.sample_dir, exist_ok=True)

    SPOT_X_COL = "pixel_x"
    SPOT_Y_COL = "pixel_y"
    CELL_TYPE_COL = "cell_type"
    CELL_SPATIAL_KEY = "spatial"
    CELL_SPATIAL_IS_YX = True   
    SCRNA_TYPE_COL = "cell_type"

    N_IMG_PCS = 32             
    RING_RADII = [args.spot_radius * 0.5, args.spot_radius * 1.0, args.spot_radius * 2.0]
    EPS = 1e-8

    seed_everything(args.seed)

    print(f"Loading data from: {args.sample_dir}")
    st_adata = ad.read_h5ad(args.st_h5ad)
    cell_adata = ad.read_h5ad(args.cell_h5ad)
    sc_adata = ad.read_h5ad(args.scrna_h5ad)

    sc_adata.var_names_make_unique()
    common_genes = st_adata.var_names.intersection(sc_adata.var_names)
    st_adata = st_adata[:, common_genes].copy()
    sc_adata = sc_adata[:, common_genes].copy()
    print(f"   Common genes: {len(common_genes)}")

    if CELL_SPATIAL_IS_YX:
        cell_coords = cell_adata.obsm[CELL_SPATIAL_KEY][:, [1, 0]]
    else:
        cell_coords = cell_adata.obsm[CELL_SPATIAL_KEY]
        
    spot_coords = st_adata.obs[[SPOT_X_COL, SPOT_Y_COL]].values

    cell_types = cell_adata.obs[CELL_TYPE_COL].astype(str).values
    unique_types = np.unique(cell_types)
    type_map = {t: i for i, t in enumerate(unique_types)}
    cell_type_indices = np.array([type_map[t] for t in cell_types])
    n_types = len(unique_types)
    n_cells = len(cell_adata)
    n_spots = len(st_adata)

    if sparse_is_sparse := sp.issparse(st_adata.X):
        st_X = st_adata.X.toarray()
    else:
        st_X = st_adata.X

    sc.pp.normalize_total(st_adata, target_sum=1e4)
    sc.pp.log1p(st_adata)

    Y_spot_program = st_adata.X.copy()

    if sp.issparse(sc_adata.X):
        sc_X_lin = np.expm1(sc_adata.X.toarray())
    else:
        sc_X_lin = np.expm1(sc_adata.X)

    phi_ref = np.zeros((n_types, len(common_genes)))
    for t_name, t_idx in type_map.items():
        t_mask = sc_adata.obs[SCRNA_TYPE_COL] == t_name
        if np.sum(t_mask) > 0:
            phi_ref[t_idx] = np.mean(sc_X_lin[t_mask], axis=0)
        else:
            print(f"Warning: Cell type {t_name} not found in scRNA-seq")

    R_mat, nearest_spot_idx, valid_dist_mask = get_R_matrix(cell_coords, spot_coords, args.spot_radius)
    N_st = np.zeros((n_spots, n_types))
    for s_idx, t_idx in zip(nearest_spot_idx[valid_dist_mask], cell_type_indices[valid_dist_mask]):
        N_st[s_idx, t_idx] += 1

    expected_expr = N_st @ phi_ref  
    denominator = expected_expr + EPS
    n_genes = st_adata.n_vars

    print("   Calculating Prior...")
    y_prior = np.zeros((n_cells, n_genes), dtype=np.float32)
    raw_st_X = np.expm1(st_adata.X.toarray()) if sp.issparse(st_adata.X) else np.expm1(st_adata.X)
    ratio_mat = raw_st_X / denominator 

    for i in range(n_cells):
        if not valid_dist_mask[i]:
            continue 
        s_id = nearest_spot_idx[i]
        t_id = cell_type_indices[i]
        y_hat_linear = ratio_mat[s_id] * phi_ref[t_id]
        y_prior[i] = np.log1p(y_hat_linear)

    print("   Initializing Alpha...")
    alpha_init = np.zeros((n_types, n_genes), dtype=np.float32)
    for t_id in range(n_types):
        mask = (cell_type_indices == t_id) & valid_dist_mask
        if np.any(mask):
            alpha_init[t_id] = np.mean(y_prior[mask], axis=0)
        else:
            alpha_init[t_id] = np.log1p(phi_ref[t_id])
        
    h_img = process_image_features(args.feat_hipt, args.feat_resccl, n_components=N_IMG_PCS)

    L_sparse, L_debug = build_typewise_blockdiag_laplacian_sparse_sym(
        cell_coords=cell_coords,
        h_img=h_img,
        cell_type_indices=cell_type_indices,
        k_spatial=100,         
        k_feature=15,
        eps=EPS,
        device=DEVICE,
    )

    e_features_raw = compute_env_features(cell_coords, cell_type_indices, RING_RADII, n_types)

    scaler_e = StandardScaler()
    e_features = scaler_e.fit_transform(e_features_raw)
    n_env_dim = e_features.shape[1]

    alpha_init_safe = alpha_init + 1e-4   
    model = Structure_Model(n_genes, n_types, n_env_dim, alpha_init_safe).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': [model.alpha], 'weight_decay': 0.0},
        {'params': [model.beta],  'weight_decay': 1e-2}
    ], lr=args.lr)

    tensor_y_spot = to_torch(_to_dense(Y_spot_program)) 
    tensor_y_prior = to_torch(y_prior)   
    tensor_e_feats = to_torch(e_features)    
    tensor_type_indices = to_torch(cell_type_indices, dtype=torch.long)

    tensor_R = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack(R_mat.nonzero()), dtype=torch.long),
        values=torch.tensor(R_mat.data, dtype=torch.float32),
        size=R_mat.shape
    ).to(DEVICE)
    tensor_prior_mask = to_torch(valid_dist_mask, dtype=torch.bool)

    criterion = torch.nn.SmoothL1Loss() 

    prev_loss = float('inf')
    no_improve_epochs = 0
    PATIENCE = 5      
    TOLERANCE = 1e-4   

    print("Starting Model Training...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        y_pred, be = model(tensor_type_indices, tensor_e_feats)
        exp_y_pred = torch.expm1(y_pred)
        y_spot_pred = torch.sparse.mm(tensor_R, exp_y_pred) 
        y_spot_pred_log = torch.log1p(y_spot_pred) 
        loss_spot = criterion(y_spot_pred_log, tensor_y_spot)
        
        if tensor_prior_mask.sum() > 0:
            loss_prior = criterion(y_pred[tensor_prior_mask], tensor_y_prior[tensor_prior_mask])
        else:
            loss_prior = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            
        loss = loss_spot + loss_prior 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        delta = abs(prev_loss - current_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={current_loss:.6f} (Delta={delta:.6f}) "
                  f"(Spot={loss_spot.item():.4f}, Prior={loss_prior.item():.4f})")
                  
        if delta < TOLERANCE:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"Converged at Epoch {epoch}! Loss didn't change > {TOLERANCE} for {PATIENCE} epochs.")
                break
        else:
            no_improve_epochs = 0 
        prev_loss = current_loss

    model.eval()
    with torch.no_grad():
        y_base_final, be_final = model(tensor_type_indices, tensor_e_feats)

    L_scipy = torch_sparse_to_scipy(L_sparse) 
    mask_np = tensor_prior_mask.cpu().numpy()
    r_target = (tensor_y_prior - y_base_final)[tensor_prior_mask].detach().cpu().numpy()
    
    r_harmonic_np = solve_harmonic_extension(L_scipy, mask_np, r_target)
    r_init_tensor = torch.from_numpy(r_harmonic_np).float().to(y_base_final.device)

    r_final_tensor = optimize_residuals_post_hoc(
        y_base_fixed=y_base_final,
        y_prior=tensor_y_prior,
        mask=tensor_prior_mask,
        tensor_R=tensor_R, 
        tensor_y_spot=tensor_y_spot,
        L_sparse=L_sparse, 
        lam=args.lambda_morph,
        lam_gamma=args.lambda_gamma,
        steps=100,
        r_init=r_init_tensor
    )

    y_total_prog = y_base_final + r_final_tensor
    y_prog_np = y_total_prog.cpu().numpy()
    be_prog_np = be_final.cpu().numpy()
    res_prog_np = r_final_tensor.cpu().numpy()
    alpha_np = model.alpha.detach().cpu().numpy()[cell_type_indices]

    y_gene_counts = np.expm1(y_prog_np)
    y_gene_counts[y_gene_counts < 0] = 0

    adata_out = ad.AnnData(X=y_gene_counts)
    adata_out.obs = cell_adata.obs.copy()
    adata_out.var_names = st_adata.var_names
    adata_out.obsm[CELL_SPATIAL_KEY] = cell_adata.obsm[CELL_SPATIAL_KEY]

    adata_out.obsm['X_pred']  = y_prog_np
    adata_out.obsm['X_alpha'] = alpha_np
    adata_out.obsm['X_niche'] = be_prog_np  
    adata_out.obsm['X_resid'] = res_prog_np 

    beta_val = model.beta.detach().cpu().numpy()
    adata_out.uns['beta_niche_weights'] = beta_val
    print("Saved Beta (Niche Weights)")

    adata_out.obsm["X_env_raw"] = e_features_raw.astype(np.float32)
    adata_out.obsm["X_env"]     = e_features.astype(np.float32)
    adata_out.obs["cell_type_indices"] = cell_type_indices

    type_names = list(unique_types)  
    env_feature_names = [f"ring{r_idx+1}_{t}" for r_idx in range(len(RING_RADII)) for t in type_names]

    adata_out.uns["X_env_feature_names"] = env_feature_names
    adata_out.uns["X_env_radii"] = list(map(float, RING_RADII))

    # 保存最终重构文件至 ./data/tissue_name/cellular_unit_result.h5ad
    save_path = os.path.join(args.sample_dir, args.save_name)
    adata_out.write_h5ad(save_path)
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    main()




