
import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
import argparse

from .identity_reconstruction_utils import (
    train_val_decoder, 
    predict_cell_type_proportions,
    nuclei_within_spot, 
    get_cell_type_fraction, 
    build_A_nonoverlap_tiling_with_offset, 
    build_four_A_mats,
    solve_spot_matrix_L1_gurobi, 
    suggest_weights, 
    solve_cells_with_multi_tilings_and_spots,
    sparsify_and_normalize_predictions,
    load_and_fuse_features
)



def parse_args():
    parser = argparse.ArgumentParser(description="Cell-identity reconstruction")
    
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--tissue_name", type=str, default="tissue_name", help="Name of the tissue sample folder")
    parser.add_argument("--st_filename", type=str, default="ST_adata.h5ad", help="ST h5ad filename")

    parser.add_argument("--num_spots_per_batch", type=int, default=4, help="Number of spots per batch for sampler")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for DecoderMLP")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--spot_radius", type=float, default=129.4, help="Radius of a spot")
    parser.add_argument("--patch_size", type=int, default=260, help="Patch size for non-overlap tiling")
    
    args = parser.parse_args()  
    
    args.sample_dir = os.path.join(args.data_dir, args.tissue_name)
    
    args.spot_fractions_path = os.path.join(args.sample_dir, "deconv_results.csv")
    args.cell_coords_path = os.path.join(args.sample_dir, "cell_segmented_coords.csv")
    args.st_h5ad_path = os.path.join(args.sample_dir, args.st_filename)
    args.vit_sub_path = os.path.join(args.sample_dir, "hipt_sub_feats.npy")
    args.vit_cls_path = os.path.join(args.sample_dir, "hipt_cls_feats.npy")
    args.cnn_feat_path = os.path.join(args.sample_dir, "retccl_feats.npy")
    
    return args

    


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Processing sample inside: {args.sample_dir}")

    cell_type_fractions_spot = pd.read_csv(args.spot_fractions_path, index_col=0)
    unit_coords = pd.read_csv(args.cell_coords_path)
    
    st_adata = ad.read_h5ad(args.st_h5ad_path)
    spot_loc = st_adata.obs.copy()
    cell_type_fractions_spot.index=cell_type_fractions_spot.index.astype(str)
    spot_loc_se = spot_loc.loc[cell_type_fractions_spot.index]

    Assign = nuclei_within_spot(unit_coords, spot_loc_se, radius=args.spot_radius)
    Assign_df = pd.DataFrame(Assign, columns=cell_type_fractions_spot.index)
    Assign_inspot = Assign_df[Assign_df.sum(axis=1) == 1]

    Assign_cells_spots = Assign_inspot.values.astype(np.float32)
    mask_spot = (Assign_cells_spots.sum(axis=0) > 0)
    valid_spots = np.where(mask_spot)[0]
    old2new = {int(old): i for i, old in enumerate(valid_spots)}

    cell_spot_old = np.where(Assign_cells_spots.sum(axis=1) > 0, Assign_cells_spots.argmax(axis=1), -1).astype(int)
    cell_spot_ids = np.array([old2new.get(int(s), -1) for s in cell_spot_old], dtype=int)

    keep_cells = (cell_spot_ids != -1)
    cell_spot_ids = cell_spot_ids[keep_cells]
    Assign_full_np = Assign_cells_spots[keep_cells][:, mask_spot].T
    
    spot_prior = cell_type_fractions_spot.values.astype(float)
    row_sums = spot_prior.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        bad = np.where(row_sums.ravel() <= 0)[0]
        raise ValueError(
            f"Deconvolution contains {len(bad)} spots with zero total proportion."
        )

    spot_prior_np = (spot_prior / row_sums)[mask_spot]

    feat_process_df = load_and_fuse_features(
        vit_sub_path=args.vit_sub_path,
        vit_cls_path=args.vit_cls_path,
        cnn_feat_path=args.cnn_feat_path
    )
    
    X_features_inspot = feat_process_df.loc[Assign_inspot.index.values]                  
    X_features = X_features_inspot.values[keep_cells]                     
    
    print("Training MLP Decoder...")
    model = train_val_decoder(
        X_features, cell_spot_ids, Assign_full_np, spot_prior_np,
        num_spots_per_batch=args.num_spots_per_batch, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, hidden=args.hidden_dim, dropout=args.dropout, device=device,
        early_stop_patience=args.early_stop_patience
    )

    coords = unit_coords[['pixel_y', 'pixel_x']].values
    cell_pred = predict_cell_type_proportions(model, feat_process_df, coords, device=device)    
    cell_pred_df = sparsify_and_normalize_predictions(cell_pred)

    type_count = Assign_inspot.sum().values.reshape(-1, 1) * cell_type_fractions_spot.round(2).values
    cellfrac = type_count.sum(axis=0) / (type_count.sum() + 1e-10)
    cellfrac_df = pd.DataFrame(np.reshape(cellfrac, (1, len(cellfrac))), index=['Fraction'])
    
    cell_type_numbers_int = get_cell_type_fraction(Assign_inspot.shape[0], cellfrac_df)
    known_type_counts = {cell_type_fractions_spot.columns.values[i]: cell_type_numbers_int[i] for i in range(len(cell_type_numbers_int))}
    
    assignment = solve_spot_matrix_L1_gurobi(Assign_inspot.sum().values, cell_type_fractions_spot, known_type_counts)
    assignment_df = pd.DataFrame(
        assignment,
        index=cell_type_fractions_spot.index, 
        columns=cell_type_fractions_spot.columns
    )
    
    cell_type_numbers_int_all = get_cell_type_fraction(Assign_df.shape[0], cellfrac_df)
    known_type_counts_all = {cell_type_fractions_spot.columns.values[i]: cell_type_numbers_int_all[i] for i in range(len(cell_type_numbers_int_all))}

    W, H = float(unit_coords['pixel_x'].max()), float(unit_coords['pixel_y'].max())
    p_ik = cell_pred_df.iloc[:, :-2].values
    
    result = build_four_A_mats(
        cell_coords=coords, img_width=W, img_height=H, p_ik=p_ik, 
        known_type_counts=known_type_counts_all, patch_size=args.patch_size
    )

    dfs = [res['df'] for res in result]
    type_names = [c for c in dfs[0].columns if c not in ("is_260_square", "y0", "x0", "y1", "x1")]

    A_list, P_list = [], []
    offsets = [(0,0), (130,0), (0,130), (130,130)]
    for (off, df) in zip(offsets, dfs):
        A_all, meta = build_A_nonoverlap_tiling_with_offset(
            cell_coords=coords, img_width=W, img_height=H,
            patch_size=args.patch_size, offset=off, return_sparse=True
        )
        boxes = meta["boxes"]             
        key_meta = pd.DataFrame(boxes, columns=["y0","x0","y1","x1"])
        key_meta["row"] = np.arange(boxes.shape[0])
        
        df_full = df[df["is_260_square"] == True].copy()
        df_key  = df_full.merge(key_meta, on=["y0","x0","y1","x1"], how="inner")
        assert len(df_key) > 0, "Alignment failed."
        
        rows = df_key["row"].to_numpy()
        A_t  = A_all.tocsr()[rows, :]         
        P_t  = df_key[type_names].to_numpy()  
        A_list.append(A_t)
        P_list.append(P_t)
        
    weights = suggest_weights(
        A_list, P_list,
        B_spot = Assign_df.T.values.astype(np.uint8), T_spot=assignment_df.values,
        p_prior=p_ik
    )
     
    X_sol, _ = solve_cells_with_multi_tilings_and_spots(
        p_prior=p_ik, A_list=A_list, P_list=P_list,
        B_spot=Assign_df.T.values.astype(np.uint8), T_spot=assignment_df.values, known_type_counts=known_type_counts_all,
        w_patch=weights["w_patch"], w_spot=weights["w_spot"], w_prior=weights["w_prior"]
    )
     
    final_labels = np.array(type_names)[X_sol.argmax(axis=1)]
    cell_ids = pd.Index([f"cell_{i}" for i in range(X_sol.shape[0])], name="cell_id")
    
    adata = ad.AnnData(
        X=np.zeros((X_sol.shape[0], 0), dtype=np.float32),              
        obs=pd.DataFrame({"cell_type": pd.Categorical(final_labels)}, index=cell_ids),
        obsm={"spatial": coords}
    )
    
    # 结果导出至 ./data/tissue_name/cellular_scaffold.h5ad
    out_h5ad_path = os.path.join(args.sample_dir, "cellular_scaffold.h5ad")
    adata.write(out_h5ad_path)
    print(f"Done! Model results exported successfully to {out_h5ad_path}")

if __name__ == "__main__":
    main()





    