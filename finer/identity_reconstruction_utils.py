
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd

import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad 


from scipy.sparse import issparse, csr_matrix, coo_matrix
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Any, List, Optional, Tuple, Union

from sklearn.preprocessing import StandardScaler, Normalizer
from scipy.spatial import cKDTree


class CellsFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        assert self.X.ndim == 2
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], idx



def build_spot_buckets(cell_spot_ids, num_spots):
    buckets = [[] for _ in range(num_spots)]
    for i, s in enumerate(cell_spot_ids):
        if 0 <= s < num_spots:
            buckets[s].append(i)
    return buckets  

class SpotCountBatchSampler(Sampler[List[int]]):
    def __init__(self, spot_buckets: List[List[int]], num_spots_per_batch: int = 4, shuffle: bool = True):
        assert num_spots_per_batch >= 1
        self.buckets = spot_buckets
        self.k = num_spots_per_batch
        self.shuffle = shuffle
    def __iter__(self):
        idxs = list(range(len(self.buckets)))
        if self.shuffle: random.shuffle(idxs)
        for i in range(0, len(idxs), self.k):
            chosen = idxs[i:i+self.k]
            batch_cells = []
            for ci in chosen: batch_cells.extend(self.buckets[ci])
            if batch_cells: yield batch_cells
    def __len__(self):
        return max(1, math.ceil(len(self.buckets)/self.k))


def collate_with_B(batch, B_full: torch.Tensor, spot_prior: torch.Tensor):
    feats, idxs = zip(*batch)
    X_b = torch.stack(feats, dim=0)             # [N_b,F]
    cols = torch.tensor(idxs, dtype=torch.long)
    B_cols = B_full[:, cols]                     # [S_all, N_b]
    row_mask = (B_cols.sum(dim=1) > 0)           
    B_sub = B_cols[row_mask]                     # [S_sel, N_b]
    spot_prior_sub = spot_prior[row_mask]        # [S_sel, C]
    return X_b, B_sub, spot_prior_sub



class DecoderMLP(nn.Module):
    """F -> C"""
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):                # x: [N_b,F]
        logits = self.net(x)             # [N_b,C]
        probs  = logits.softmax(dim=-1)  # [N_b,C]
        return logits, probs



def aggregate_spot_with_B(cell_probs: torch.Tensor, B_sub: torch.Tensor, eps: float = 1e-6):
    """
    cell_probs: [N_b,C]；B_sub: [S_sel,N_b]
    """
    S_sel, N_b = B_sub.shape
    assert cell_probs.shape[0] == N_b
    C = cell_probs.shape[1]
    num = B_sub @ cell_probs                                # [S_sel, C]
    den = B_sub.sum(dim=1, keepdim=True).clamp_min(1.0)     # [S_sel, 1]
    spot_pred = num / den                                   
    spot_pred = (spot_pred + eps) / (spot_pred.sum(dim=1, keepdim=True) + C*eps)
    return spot_pred


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def _norm(x):
        return (x + eps) / (x.sum(dim=1, keepdim=True) + x.shape[1] * eps)

    p, q = _norm(p), _norm(q)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=1)
    kl_qm = (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm).mean()




def train_val_decoder(
    X, cell_spot_ids, B_full_np, spot_prior_np,
    num_spots_per_batch=4, lr=2e-4, weight_decay=1e-4,
    epochs=10, hidden=512, dropout=0.2, device="cuda",
    val_ratio=0.2, seed=42, early_stop_patience=None
):
    X = np.asarray(X, dtype=np.float32)
    N, F_dim = X.shape
    B_full = torch.as_tensor(B_full_np, dtype=torch.float32, device=device)   # [S,N]
    spot_prior = torch.as_tensor(spot_prior_np, dtype=torch.float32, device=device) # [S,C]
    S, C = spot_prior.shape
    rng = np.random.default_rng(seed)
    all_spots = np.arange(S)
    rng.shuffle(all_spots)
    n_val = max(1, int(S * val_ratio))
    val_spots = set(all_spots[:n_val])
    train_spots = set(all_spots[n_val:])
    ds = CellsFeatureDataset(X)   
    buckets_all = build_spot_buckets(cell_spot_ids, S)  # list[list[int]], len = S
    train_ids = [s for s in range(S) if (s in train_spots) and (len(buckets_all[s])>0)]
    val_ids   = [s for s in range(S) if (s in val_spots)   and (len(buckets_all[s])>0)]
    buckets_tr = [buckets_all[s] for s in train_ids]   # list[list[int]]
    buckets_va = [buckets_all[s] for s in val_ids]     # list[list[int]]
    train_sampler = SpotCountBatchSampler(buckets_tr, num_spots_per_batch=num_spots_per_batch, shuffle=True)
    val_sampler   = SpotCountBatchSampler(buckets_va,   num_spots_per_batch=min(num_spots_per_batch, max(1, len(buckets_va))), shuffle=False)
    def mk_loader(sampler):
        def collate_fn(batch):
            X_b, B_sub, spot_prior_sub = collate_with_B(batch, B_full, spot_prior)
            return X_b.to(device, non_blocking=True), B_sub.to(device), spot_prior_sub.to(device)
        return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn)
    dl_tr = mk_loader(train_sampler)
    dl_va = mk_loader(val_sampler)
    model = DecoderMLP(in_dim=F_dim, num_classes=C, hidden=hidden, p=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    patience = early_stop_patience
    stall = 0
    for ep in range(1, epochs+1):
        model.train()
        tr_loss_sum, tr_steps = 0.0, 0
        for X_b, B_sub, spot_prior_sub in dl_tr:
            _, probs = model(X_b)                    # [N_b, C]
            spot_pred = aggregate_spot_with_B(probs, B_sub)
            loss = js_divergence(spot_prior_sub, spot_pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss_sum += float(loss.item()); tr_steps += 1
        model.eval()
        with torch.no_grad():
            va_loss_sum, va_steps = 0.0, 0
            for X_b, B_sub, spot_prior_sub in dl_va:
                _, probs = model(X_b)
                spot_pred = aggregate_spot_with_B(probs, B_sub)
                loss = js_divergence(spot_prior_sub, spot_pred)
                va_loss_sum += float(loss.item()); va_steps += 1
        tr_loss = tr_loss_sum / max(1, tr_steps)
        va_loss = va_loss_sum / max(1, va_steps)
        print(f"[Epoch {ep}] train_JS={tr_loss:.4f} | val_JS={va_loss:.4f} (spots: train={len(buckets_tr)}, val={len(buckets_va)})")
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if patience is not None and stall >= patience:
                print(f"Early stop at epoch {ep} (best val_KL={best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model



def predict_cell_type_proportions(model, X, coordinates, device="cuda"):
    model.eval()  
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    X_tensor = torch.from_numpy(X).to(device)

    with torch.no_grad():
        # (logits, probs)
        _, probs = model(X_tensor)
    preds_df = pd.DataFrame(probs.cpu().numpy(), columns=[f"Type_{i}" for i in range(probs.shape[1])])
    preds_df[['y', 'x']] = coordinates
    return preds_df



def sparsify_and_normalize_predictions(
    df: pd.DataFrame,
    threshold_ratio: float = 0.2,
    coord_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:

    df_out = df.copy()

    if coord_cols is None:
        if 'x' in df_out.columns and 'y' in df_out.columns:
            coord_cols = ['y', 'x']
        else:
            coord_cols = list(df_out.columns[-2:])  

    feat_cols = [col for col in df_out.columns if col not in coord_cols]

    thresholds = df_out[feat_cols].max() * threshold_ratio
    df_out[feat_cols] = df_out[feat_cols].mask(df_out[feat_cols] < thresholds, 0.0)
    row_sums = df_out[feat_cols].sum(axis=1).replace(0, np.nan)
    df_out[feat_cols] = df_out[feat_cols].div(row_sums, axis=0).fillna(0.0)

    return df_out     




def nuclei_within_spot(srspots, spot_loc, radius):
    coords = srspots[['pixel_y', 'pixel_x']].to_numpy()
    spot_centers = spot_loc[['pixel_y', 'pixel_x']].to_numpy()

    tree = cKDTree(spot_centers)
    neighbors = tree.query_ball_point(coords, r=radius)

    B = np.zeros((len(coords), len(spot_centers)), dtype=np.float32)

    for i, ids in enumerate(neighbors):
        if ids:
            B[i, ids] = 1.0

    return B


def get_cell_type_fraction(number_of_cells: int, cell_type_fraction_data: pd.DataFrame) -> np.ndarray:
    df_sorted = cell_type_fraction_data.reindex(sorted(cell_type_fraction_data.columns), axis=1)
    fractions = df_sorted.values[0]
    
    numbers = number_of_cells * fractions + 0.5
    numbers_int = numbers.astype(int)
    diff = number_of_cells - np.sum(numbers_int)
    
    max_idx = np.argmax(numbers)
    numbers_int[max_idx] += diff
    return numbers_int


def _make_cuts(L, patch_size, offset):
    L = float(L)
    ps = int(patch_size)
    off = int(max(0, offset))
    cuts = [0]
    if off > 0 and off < L:
        cuts.append(off)
    if off < L:
        x = off
        while x < L:
            x_next = x + ps
            if x_next >= L:
                cuts.append(L)
                break
            cuts.append(x_next)
            x = x_next
    else:
        if cuts[-1] != L:
            cuts.append(L)
    cuts = np.array(cuts, dtype=np.float64)
    cuts = np.unique(cuts)
    spans = np.diff(cuts)
    return cuts, spans



def build_A_nonoverlap_tiling_with_offset(
    cell_coords,             # (N,2) [y,x] 
    img_width, img_height,   
    patch_size=260,
    offset=(0,0),
    return_sparse=True
):
    ps = int(patch_size)
    oy, ox = int(offset[0]), int(offset[1])
    cuts_x, spans_x = _make_cuts(img_width,  ps, ox)
    cuts_y, spans_y = _make_cuts(img_height, ps, oy)
    nx = len(spans_x)
    ny = len(spans_y)
    S  = nx * ny
    coords = np.asarray(cell_coords, dtype=np.float64)
    ys = coords[:, 0]; xs = coords[:, 1]
    N  = coords.shape[0]
    xs_clamped = np.clip(xs, 0, np.nextafter(float(img_width), -np.inf))
    ys_clamped = np.clip(ys, 0, np.nextafter(float(img_height), -np.inf))
    ix = np.searchsorted(cuts_x, xs_clamped, side='right') - 1
    iy = np.searchsorted(cuts_y, ys_clamped, side='right') - 1
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    rows = (iy * nx + ix).astype(np.int64)
    cols = np.arange(N, dtype=np.int64)
    if return_sparse:
        data = np.ones(N, dtype=np.uint8)
        A_ri = coo_matrix((data, (rows, cols)), shape=(S, N))
    else:
        A_ri = np.zeros((S, N), dtype=np.uint8)
        A_ri[rows, cols] = 1
    x0s = cuts_x[:-1]; x1s = cuts_x[1:]
    y0s = cuts_y[:-1]; y1s = cuts_y[1:]
    boxes = np.zeros((S, 4), dtype=np.int64)
    full_square_mask = np.zeros(S, dtype=bool)
    widths  = (x1s - x0s)            # (nx,)
    heights = (y1s - y0s)            # (ny,)
    full_grid = ((heights[:, None] == ps) & (widths[None, :] == ps))  # (ny, nx)
    r = 0
    for iy0 in range(ny):
        y0 = y0s[iy0]; y1 = y1s[iy0]
        for ix0 in range(nx):
            x0 = x0s[ix0]; x1 = x1s[ix0]
            boxes[r] = (int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
            full_square_mask[r] = bool(full_grid[iy0, ix0])
            r += 1
    meta = dict(
        cuts_x=cuts_x, cuts_y=cuts_y,
        spans_x=spans_x, spans_y=spans_y,
        boxes=boxes,
        full_square_mask=full_square_mask,
        nx=nx, ny=ny, total_patches=S,
        patch_size=ps, offset=(oy, ox)
    )
    return A_ri, meta



def build_four_A_mats(cell_coords, img_width, img_height, p_ik, known_type_counts, patch_size=260):
    offsets = [(0,0), (130,0), (0,130), (130,130)]
    results = []
    for off in offsets:
        A, meta = build_A_nonoverlap_tiling_with_offset(
            cell_coords=cell_coords,
            img_width=img_width, img_height=img_height,
            patch_size=patch_size, offset=off, return_sparse=False
        )
        X = solve_integer_matrix_L1_gurobi(
        p_ik=p_ik, A_ri=A, known_type_counts = known_type_counts)
        full_mask = meta["full_square_mask"]       # (S,)
        boxes     = meta["boxes"]                  # (S,4): (y0,x0,y1,x1)
        df = pd.DataFrame(X, columns = list(known_type_counts.keys()))
        df["is_260_square"] = full_mask.astype(bool)
        df["y0"], df["x0"], df["y1"], df["x1"] = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        results.append(dict(offset=off, A=A, meta=meta, X=X, df=df))
    return results





def solve_integer_matrix_L1_gurobi(p_ik, A_ri, known_type_counts):
    b_rk = np.dot(A_ri, p_ik)
    M, K = b_rk.shape
    type_names = list(known_type_counts.keys())
    m = gp.Model("SpotTypeInteger_L1")
    m.Params.OutputFlag = 1
    x = m.addVars(M, K, vtype=GRB.INTEGER, lb=0, name="x")
    dplus  = m.addVars(M, K, vtype=GRB.CONTINUOUS, lb=0.0, name="dplus")
    dminus = m.addVars(M, K, vtype=GRB.CONTINUOUS, lb=0.0, name="dminus")
    for r in range(M):
        for k in range(K):
            m.addConstr(x[r, k] - b_rk[r, k] == dplus[r, k] - dminus[r, k])
    for r in range(M):
        m.addConstr(gp.quicksum(x[r, k] for k in range(K)) == int(np.sum(A_ri[r, :])))
    for k, name in enumerate(type_names):
        m.addConstr(gp.quicksum(x[r, k] for r in range(M)) == int(round(known_type_counts[name])))
    m.setObjective(gp.quicksum(dplus.values()) + gp.quicksum(dminus.values()), GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Solver status: {m.Status}")
    X = np.zeros((M, K), dtype=int)
    for r in range(M):
        for k in range(K):
            X[r, k] = int(round(x[r, k].X))
    return X



def suggest_weights(
    A_list, P_list,
    B_spot=None, T_spot=None,
    p_prior=None,
    *,
    C=1e5,          
    eps=1e-6,       
    cap=20.0,      
    ratio_patch=1.0, ratio_spot=1.0, ratio_prior=1.0  
):
    N = None
    K = None
    S_patch_total = 0
    patch_row_scale = []  
    for t, (A_t, P_t) in enumerate(zip(A_list, P_list)):
        if N is None: N = A_t.shape[1]
        if K is None: K = P_t.shape[1]
        assert A_t.shape[1] == N, f"A_list[{t}] column count mismatch with N"
        assert P_t.shape[1] == K, f"P_list[{t}] column count mismatch with K"
        S_t = P_t.shape[0]
        S_patch_total += S_t
        row_sum = np.asarray(P_t).sum(axis=1)   
        patch_row_scale.append(row_sum)
    if len(patch_row_scale) > 0:
        patch_row_scale = np.concatenate(patch_row_scale, axis=0)
        s_patch = float(np.median(patch_row_scale)) if patch_row_scale.size > 0 else 1.0
        if s_patch <= 0: s_patch = 1.0
    else:
        s_patch = 1.0
        S_patch_total = 0
    S_spot = 0
    s_spot = 1.0
    if B_spot is not None and T_spot is not None:
        assert T_spot.shape[1] == K
        S_spot = T_spot.shape[0]
        spot_row_sum = np.asarray(T_spot).sum(axis=1)
        s_spot = float(np.median(spot_row_sum)) if spot_row_sum.size > 0 else 1.0
        if s_spot <= 0: s_spot = 1.0
    mu_prior = 1.0
    if p_prior is not None:
        assert p_prior.shape[0] == N and p_prior.shape[1] == K
        p_safe = np.clip(p_prior, eps, 1.0)
        mx = p_safe.max(axis=1)
        mu_prior = float(np.median(-np.log(mx)))
        if not np.isfinite(mu_prior) or mu_prior <= 0:
            mu_prior = 1.0
        mu_prior = min(mu_prior, cap)
    w_patch = (C / (S_patch_total * s_patch)) if (S_patch_total > 0) else 0.0
    w_spot  = (C / (S_spot       * s_spot )) if (S_spot        > 0) else 0.0
    w_prior = (C / (N            * mu_prior)) if (p_prior is not None) else 0.0
    w_patch *= float(ratio_patch)
    w_spot  *= float(ratio_spot)
    w_prior *= float(ratio_prior)
    if not np.isfinite(w_patch) or w_patch < 0:
        w_patch = 0.0
    if not np.isfinite(w_spot) or w_spot < 0:
        w_spot = 0.0
    if not np.isfinite(w_prior) or w_prior < 0:
        w_prior = 0.0
    diag = {
        "S_patch_total": int(S_patch_total),
        "s_patch_median": float(s_patch),
        "S_spot": int(S_spot),
        "s_spot_median": float(s_spot),
        "mu_prior_median_neglog_maxp": float(mu_prior),
        "C": float(C),
        "eps": float(eps),
        "cap": float(cap),
    }
    return {"w_patch": w_patch, "w_spot": w_spot, "w_prior": w_prior, "diag": diag}


def solve_spot_matrix_L1_gurobi(spot_cells_count, type_proportions, known_type_counts,
                                   time_limit=None, mip_gap=None):
    S = len(spot_cells_count)
    type_names = list(type_proportions.columns)
    Tn = len(type_names)
    T = type_proportions.round(2).values * spot_cells_count.reshape(-1, 1)  # [S,T]
    if int(np.round(spot_cells_count.sum())) != int(np.round(sum(known_type_counts[n] for n in type_names))):
        raise ValueError("Mismatched total cells: sum(row_sums) is not equal to sum(col_sums)")
    m = gp.Model("SpotTypeInteger_L1")
    m.Params.OutputFlag = 1
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    x = m.addVars(S, Tn, vtype=GRB.INTEGER, lb=0, name="x")
    dplus  = m.addVars(S, Tn, vtype=GRB.CONTINUOUS, lb=0.0, name="dplus")
    dminus = m.addVars(S, Tn, vtype=GRB.CONTINUOUS, lb=0.0, name="dminus")
    for s in range(S):
        for t in range(Tn):
            m.addConstr(x[s, t] - T[s, t] == dplus[s, t] - dminus[s, t])
    for s in range(S):
        m.addConstr(gp.quicksum(x[s, t] for t in range(Tn)) == int(round(spot_cells_count[s])))
    for t, name in enumerate(type_names):
        m.addConstr(gp.quicksum(x[s, t] for s in range(S)) == int(round(known_type_counts[name])))
    m.setObjective(gp.quicksum(dplus.values()) + gp.quicksum(dminus.values()), GRB.MINIMIZE)
    m.Params.MIPFocus = 1         
    m.Params.Heuristics = 0.5
    m.Params.NumericFocus = 1
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Solver status: {m.Status}")
    X = np.zeros((S, Tn), dtype=int)
    for s in range(S):
        for t in range(Tn):
            X[s, t] = int(round(x[s, t].X))
    return X


def solve_cells_with_multi_tilings_and_spots(
    p_prior,                 # (N,K) or None
    A_list,                  # list of (S_t x N)
    P_list,                  # list of (S_t x K)
    B_spot=None,             # (S_s x N) or None
    T_spot=None,             # (S_s x K) or None
    *,
    known_type_counts=None, 
    w_patch=1.0,
    w_spot=1.0,
    w_prior=0.0,
    time_limit=3600, mip_gap=0.02, mipfocus=1, threads=8, verbose=True
):
    assert len(A_list) == len(P_list) >= 1, "A_list and P_list must have equal length ≥ 1"
    N = A_list[0].shape[1]
    K = P_list[0].shape[1]
    for t,(A_t,P_t) in enumerate(zip(A_list, P_list)):
        assert A_t.shape[1] == N, f"A_list[{t}] column count mismatch N"
        assert P_t.shape[1] == K, f"P_list[{t}] column count mismatch K"
    if B_spot is not None:
        assert T_spot is not None, "T_spot required when B_spot exists"
        assert B_spot.shape[1] == N and T_spot.shape[1] == K, "B_spot/T_spot shape mismatch"
    # 处理先验
    if p_prior is not None and w_prior > 0:
        assert p_prior.shape == (N, K)
        logp = np.log(np.clip(p_prior, 1e-9, 1.0))
    else:
        logp = None
    type_names = list(known_type_counts.keys())
    # Gurobi
    m = gp.Model("CellsTypes_MultiTilings_Spots_L1")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None: m.Params.TimeLimit = float(time_limit)
    if mip_gap    is not None: m.Params.MIPGap    = float(mip_gap)
    if mipfocus   is not None: m.Params.MIPFocus  = int(mipfocus)
    if threads    is not None: m.Params.Threads   = int(threads)
    X = m.addVars(N, K, vtype=GRB.BINARY, name="X")
    for i in range(N):
        m.addConstr(gp.quicksum(X[i,k] for k in range(K)) == 1, name=f"onehot_{i}")
    for k, name in enumerate(type_names):
        m.addConstr(gp.quicksum(X[i,k] for i in range(N)) == int(round(known_type_counts[name])), name=f"type_total_k{k}")
    patch_dev_terms = []
    for t,(A_t,P_t) in enumerate(zip(A_list, P_list)):
        S_t = P_t.shape[0]
        dpl = m.addVars(S_t, K, vtype=GRB.CONTINUOUS, lb=0.0, name=f"dplus_t{t}")
        dmn = m.addVars(S_t, K, vtype=GRB.CONTINUOUS, lb=0.0, name=f"dminus_t{t}")
        A_csr = A_t.tocsr() if issparse(A_t) else csr_matrix(A_t)
        A_csr.sort_indices()
        for r in range(S_t):
            row = A_csr.getrow(r)
            idx_i = row.indices
            coeff = row.data
            for k in range(K):
                m.addConstr(
                    gp.quicksum(float(coeff[j]) * X[int(idx_i[j]), k] for j in range(len(idx_i)))
                    - float(P_t[r, k]) == dpl[r, k] - dmn[r, k],
                    name=f"patch_bal_t{t}_r{r}_k{k}"
                )
        patch_dev_terms.append(gp.quicksum(dpl.values()) + gp.quicksum(dmn.values()))
    spot_dev_term = 0
    if B_spot is not None:
        B_csr = B_spot.tocsr() if issparse(B_spot) else csr_matrix(B_spot)
        B_csr.sort_indices()
        S_s = T_spot.shape[0]
        ds_pl = m.addVars(S_s, K, vtype=GRB.CONTINUOUS, lb=0.0, name="spot_dplus")
        ds_mn = m.addVars(S_s, K, vtype=GRB.CONTINUOUS, lb=0.0, name="spot_dminus")
        for r in range(S_s):
            row = B_csr.getrow(r)
            idx_i = row.indices
            coeff = row.data
            for k in range(K):
                m.addConstr(
                    gp.quicksum(float(coeff[j]) * X[int(idx_i[j]), k] for j in range(len(idx_i)))
                    - float(T_spot[r, k]) == ds_pl[r, k] - ds_mn[r, k],
                    name=f"spot_bal_r{r}_k{k}"
                )
        spot_dev_term = gp.quicksum(ds_pl.values()) + gp.quicksum(ds_mn.values())
    prior_term = 0
    if logp is not None and w_prior > 0:
        prior_term = gp.quicksum(- float(logp[i,k]) * X[i,k] for i in range(N) for k in range(K))
    obj = w_patch * gp.quicksum(patch_dev_terms) + w_spot * spot_dev_term + w_prior * prior_term
    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.Heuristics = 0.1
    m.optimize()
    X_sol = np.zeros((N, K), dtype=np.int8)
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        for i in range(N):
            for k in range(K):
                X_sol[i, k] = int(round(X[i, k].X))
        info = dict(status=m.Status, obj=m.ObjVal, runtime=m.Runtime, mipgap=getattr(m, "MIPGap", None))
        return X_sol, info
    else:
        raise RuntimeError(f"Gurobi status={m.Status}")
        
        
def load_and_fuse_features(
    vit_sub_path: str, 
    vit_cls_path: str, 
    cnn_feat_path: str
) -> pd.DataFrame:
   
    X1 = np.load(vit_sub_path, allow_pickle=True)
    X2 = np.load(vit_cls_path, allow_pickle=True)
    X3 = np.load(cnn_feat_path, allow_pickle=True)
    
    X1z = StandardScaler().fit_transform(X1)
    X2z = StandardScaler().fit_transform(X2)
    X3z = StandardScaler().fit_transform(X3)
    
    X1z *= 1 / np.sqrt(X1z.shape[1])
    X2z *= 1 / np.sqrt(X2z.shape[1])
    X3z *= 1 / np.sqrt(X3z.shape[1])
    
    Z = np.concatenate([X1z, X2z, X3z], axis=1)
    
    feat_process = Normalizer(norm="l2").fit_transform(Z)
    feat_process_df = pd.DataFrame(feat_process)
    
    return feat_process_df
        
        




