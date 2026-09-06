


import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
#from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import random





DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_torch(x, dtype=torch.float32, device=DEVICE):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)

def get_R_matrix(cell_coords, spot_coords, radius):
    tree = cKDTree(spot_coords)
    dists, indices = tree.query(cell_coords, k=1)
    valid_mask = dists <= radius
    row_indices = np.where(valid_mask)[0]
    col_indices = indices[valid_mask]
    data = np.ones(len(row_indices))
    R = sp.csr_matrix((data, (col_indices, row_indices)), 
                      shape=(len(spot_coords), len(cell_coords)))
    return R, indices, valid_mask


def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)




def process_image_features(hipt_path, resccl_path, n_components=30):
    df_h = np.load(hipt_path) 
    df_r = np.load(resccl_path)
    X_h = df_h.astype(np.float32)
    X_r = df_r.astype(np.float32)
    X_h = StandardScaler().fit_transform(X_h)
    X_r = StandardScaler().fit_transform(X_r)
    X_comb = np.hstack([X_h, X_r])
    pca_img = PCA(n_components=n_components)
    X_final = pca_img.fit_transform(X_comb)
    return X_final




def build_typewise_blockdiag_laplacian_sparse_sym(
    cell_coords,          # (N,2)
    h_img,                # (N,F)
    cell_type_indices,    # (N,) int
    k_spatial=50,        
    k_feature=10,
    eps=1e-8,
    device="cpu",
):
    """
    Type-wise kNN within each cell type.
    Build directed W (within type) -> symmetrize by (W+W^T)/2 -> L = D - W_sym.
    Assemble block-diagonal L into a global torch sparse tensor.

    Returns:
      L_sparse: torch.sparse_coo_tensor (N,N)
      debug: dict with per-type stats
    """
    N = cell_coords.shape[0]
    assert h_img.shape[0] == N
    assert cell_type_indices.shape[0] == N
    rows_all = []
    cols_all = []
    vals_all = []
    debug = {}
    for t in np.unique(cell_type_indices):
        idx = np.where(cell_type_indices == t)[0]
        n_t = idx.size
        if n_t <= 1:
            continue
        curr_k_spatial = int(min(k_spatial, n_t - 1))
        curr_k_feature = int(min(k_feature, curr_k_spatial))
        if curr_k_feature <= 0:
            continue
        # --- kNN inside type (use spatial coords to find neighbors) ---
        nn = NearestNeighbors(n_neighbors=curr_k_spatial + 1, metric="euclidean")
        nn.fit(cell_coords[idx])
        _, nbrs = nn.kneighbors(cell_coords[idx], return_distance=True)
        # directed edges local: i -> nbrs[i,1:]
        src_local = np.repeat(np.arange(n_t), curr_k_spatial)
        dst_local = nbrs[:, 1:].reshape(-1)
        # map to global indices
        src = idx[src_local]
        dst = idx[dst_local]
        d_feat_sq = np.sum((h_img[src] - h_img[dst]) ** 2, axis=1)
        d_feat_matrix = d_feat_sq.reshape(n_t, curr_k_spatial)
        dst_matrix = dst_local.reshape(n_t, curr_k_spatial) 
        top_k_indices = np.argpartition(d_feat_matrix, curr_k_feature - 1, axis=1)[:, :curr_k_feature]
        row_indices = np.arange(n_t)[:, None] 
        # (n_t, k_feature)
        final_d_feat = d_feat_matrix[row_indices, top_k_indices]
        # (n_t, k_feature) 
        final_dst_local = dst_matrix[row_indices, top_k_indices]
        final_d_feat_flat = final_d_feat.reshape(-1)
        final_src_flat = np.repeat(np.arange(n_t), curr_k_feature)
        final_dst_flat = final_dst_local.reshape(-1)
        sigma_f = np.median(final_d_feat_flat) + eps
        w_dir = np.exp(-final_d_feat_flat / sigma_f).astype(np.float32)
        # --- build directed sparse W in LOCAL indexing (n_t x n_t) ---
        W = sp.coo_matrix((w_dir, (final_src_flat, final_dst_flat)), shape=(n_t, n_t)).tocsr()
        # --- symmetrize by average: W_sym = (W + W^T)/2 ---
        W_sym = (W + W.T) * 0.5
        W_sym.eliminate_zeros()
        # --- Laplacian L = D - W_sym ---
        deg = np.array(W_sym.sum(axis=1)).ravel().astype(np.float32)
        L_local = sp.diags(deg, format="csr") - W_sym  # (n_t x n_t)
        # --- lift L_local back to GLOBAL indices by shifting row/col ---
        L_local = L_local.tocoo()
        rows_global = idx[L_local.row]
        cols_global = idx[L_local.col]
        vals_global = L_local.data.astype(np.float32)
        rows_all.append(rows_global.astype(np.int64))
        cols_all.append(cols_global.astype(np.int64))
        vals_all.append(vals_global)
        debug[int(t)] = {
            "n_cells": int(n_t),
            "k_spatial": curr_k_spatial, 
            "k_feature": curr_k_feature, 
            "nnz_Wsym": int(W_sym.nnz),
            "sigma_f": float(sigma_f),
        }
    if len(rows_all) == 0:
        diag = np.arange(N, dtype=np.int64)
        indices = np.vstack([diag, diag])
        values = np.ones(N, dtype=np.float32)
        L_sparse = torch.sparse_coo_tensor(
            indices=torch.tensor(indices, dtype=torch.long),
            values=torch.tensor(values, dtype=torch.float32),
            size=(N, N),
        ).coalesce().to(device)
        return L_sparse, debug
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all).astype(np.float32)
    indices = np.vstack([rows, cols]).astype(np.int64)
    L_sparse = torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long),
        values=torch.tensor(vals, dtype=torch.float32),
        size=(N, N),
    ).coalesce().to(device)
    return L_sparse, debug



def compute_env_features(coords, types, radii, n_types):
    """
    Vectorized version of compute_env_features using Sparse Matrix Multiplication.
    Much faster than iterating through lists.
    """
    N = len(coords)
    tree = cKDTree(coords)
    # rows: 0..N-1, cols: types array, values: 1
    type_onehot = np.zeros((N, n_types), dtype=np.float32)
    type_onehot[np.arange(N), types] = 1.0
    feats = []
    for r in radii:
        neighbors = tree.query_ball_point(coords, r)
        n_neighbors = np.array([len(x) for x in neighbors])
        col_indices = np.concatenate(neighbors)
        row_indices = np.repeat(np.arange(N), n_neighbors)
        data = np.ones(len(col_indices), dtype=np.float32)
        adj_mat = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
        counts = adj_mat.dot(type_onehot)
        counts -= type_onehot
        total = counts.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0 
        counts /= total
        feats.append(counts)
    return np.hstack(feats)




def seed_everything(seed=32):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class Structure_Model(nn.Module):
    def __init__(self, n_programs, n_types, n_env_dim, alpha_init):
        super().__init__()
        self.alpha = nn.Parameter(to_torch(alpha_init)) 
        self.beta = nn.Parameter(torch.randn(n_types, n_programs, n_env_dim) * 0.01)
    def forward(self, type_indices, e_feats):
        a = self.alpha[type_indices]
        be = torch.zeros_like(a)
        n_types = self.alpha.size(0)
        for c in range(n_types):
            mask = (type_indices == c)
            if mask.sum() > 0:  
                e_c = e_feats[mask]
                beta_c = self.beta[c]
                be[mask] = torch.matmul(e_c, beta_c.T)
        y_pred_raw = a + be
        y_pred = torch.clamp(y_pred_raw, min=0.0, max=20.0)
        return y_pred, be








def solve_harmonic_extension(L_sparse_scipy, mask_np, r_target_np):
    idx_in = np.where(mask_np)[0]
    idx_out = np.where(~mask_np)[0]
    if len(idx_out) == 0:
        return np.zeros((len(mask_np), r_target_np.shape[1]))
    print(f"Solving Harmonic Extension... (Out: {len(idx_out)}, In: {len(idx_in)})")
    L_out_out = L_sparse_scipy[idx_out, :][:, idx_out]
    L_out_in = L_sparse_scipy[idx_out, :][:, idx_in]
    B = -L_out_in.dot(r_target_np)
    epsilon = 1e-6
    I_eps = sp.eye(L_out_out.shape[0]) * epsilon
    L_out_out_stable = L_out_out + I_eps
    try:
        r_out = spsolve(L_out_out_stable, B)
    except Exception as e:
        print(f"Error in spsolve: {e}")
        r_out = np.zeros((len(idx_out), r_target_np.shape[1]))
    if np.isnan(r_out).any():
        print("WARNING: spsolve returned NaNs! Filling with zeros.")
        r_out = np.nan_to_num(r_out, nan=0.0)
    if r_out.ndim == 1 and r_target_np.ndim == 2:
        r_out = r_out[:, np.newaxis]
    r_full = np.zeros((L_sparse_scipy.shape[0], r_target_np.shape[1]), dtype=r_target_np.dtype)
    r_full[idx_in] = r_target_np
    r_full[idx_out] = r_out
    return r_full



def torch_sparse_to_scipy(torch_sparse_mat):
    indices = torch_sparse_mat.indices().cpu().numpy()
    values = torch_sparse_mat.values().cpu().numpy()
    shape = torch_sparse_mat.size()
    return sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)





def optimize_residuals_post_hoc(y_base_fixed, y_prior, mask, tensor_R, tensor_y_spot, L_sparse, lam, lam_gamma, r_init=None, steps=30, lr=0.01):
    diff = y_prior - y_base_fixed
    target_r = diff[mask].detach()
    if r_init is not None:
        print("Using provided r_init (Warm Start)...")
        if not isinstance(r_init, torch.Tensor):
            r_init = torch.tensor(r_init)
        r_opt = r_init.clone().detach().float().to(y_base_fixed.device)
    else:
        print("Initializing r with zeros (Cold Start)...")
        r_opt = torch.zeros_like(y_base_fixed)
    r_opt.requires_grad_(True)
    optim_r = torch.optim.Adam([r_opt], lr=lr)
    prev_loss = float('inf')
    tol = 1e-4  
    for i in range(steps):
        optim_r.zero_grad()
        Lr = torch.sparse.mm(L_sparse, r_opt)
        loss_morph = torch.mean(r_opt * Lr)
        y_total = y_base_fixed + r_opt
        y_total_safe = torch.clamp(y_total, min=0.0, max=20.0)
        cell_counts = torch.expm1(y_total_safe)
        spot_counts_pred = torch.sparse.mm(tensor_R, cell_counts)
        y_spot_pred_log = torch.log1p(spot_counts_pred + 1e-6)
        loss_spot = F.smooth_l1_loss(y_spot_pred_log, tensor_y_spot, reduction='mean')
        loss_prior_val = F.smooth_l1_loss(y_total_safe[mask], y_prior[mask], reduction='mean')  
        loss_gamma = torch.mean(r_opt ** 2)
        loss = loss_spot + loss_prior_val  + lam * loss_morph + lam_gamma * loss_gamma
        loss.backward()
        optim_r.step()
        curr_loss = loss.item()
        delta = abs(prev_loss - curr_loss) 
        if (i+1) % 10 == 0:
             print(f"  Step {i+1}: Loss={curr_loss:.6f} "
                   f"(Spot={loss_spot.item():.4f}, Prior={loss_prior_val.item():.6f}, Morph={loss_morph.item():.6f}, Gamma={loss_gamma.item():.6f})")
        if delta < tol:
            break
        prev_loss = curr_loss
    return r_opt.detach()













