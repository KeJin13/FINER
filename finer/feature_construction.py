import argparse
import os
import numpy as np
import pandas as pd


class FeatureConstruction:
    def __init__(
        self,
        save_path,
        img_path,
        tissue_name,
        spot_diameter_pixel,
        spot_diameter_um,
        checkpoint_path,
        target_pixel_size_um=0.5,
        device="cuda",
        retccl_device="cpu",
        retccl_scales_um=(20, 35, 55),
        retccl_out_dim=256,
        retccl_batch_size=128,
        csv_filename="cell_segmented_coords.csv",
    ):
        self.work_dir = os.path.join(save_path, tissue_name)
        self.img_path = img_path

        self.pixel_size_um = spot_diameter_um / spot_diameter_pixel
        self.target_pixel_size_um = target_pixel_size_um

        self.checkpoint_path = checkpoint_path
        self.device = device

        self.retccl_device = retccl_device
        self.retccl_scales_um = retccl_scales_um
        self.retccl_out_dim = retccl_out_dim
        self.retccl_batch_size = retccl_batch_size

        csv_path = os.path.join(self.work_dir, csv_filename)
        self.cell_df = pd.read_csv(csv_path)

    def run_construct(self):
        from tqdm.auto import tqdm

        from .features.hipt import hipt_features
        from .features.retccl import retccl_features

        with tqdm(
            total=3,
            desc="FINER feature construction",
            unit="stage",
        ) as progress:

            progress.set_postfix_str("HIPT")
            hipt = hipt_features(
                cell_df=self.cell_df,
                img_path=self.img_path,
                pixel_size_um=self.pixel_size_um,
                target_pixel_size_um=self.target_pixel_size_um,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            progress.update(1)

            progress.set_postfix_str("RetCCL")
            retccl = retccl_features(
                cell_df=self.cell_df,
                img_path=self.img_path,
                pixel_size_um=self.pixel_size_um,
                target_pixel_size_um=self.target_pixel_size_um,
                checkpoint_path=self.checkpoint_path,
                device=self.retccl_device,
                scales_um=self.retccl_scales_um,
                out_dim=self.retccl_out_dim,
                batch_size=self.retccl_batch_size,
                temp_dir=self.work_dir,
            )
            progress.update(1)

            progress.set_postfix_str("saving numpy features")

            hipt_sub = hipt["sub"].to_numpy() if hasattr(hipt["sub"], "to_numpy") else np.asarray(hipt["sub"])
            hipt_cls = hipt["cls"].to_numpy() if hasattr(hipt["cls"], "to_numpy") else np.asarray(hipt["cls"])
            retccl_array = retccl.to_numpy() if hasattr(retccl, "to_numpy") else np.asarray(retccl)

            np.save(os.path.join(self.work_dir, "hipt_sub_feats.npy"), hipt_sub)
            np.save(os.path.join(self.work_dir, "hipt_cls_feats.npy"), hipt_cls)
            np.save(os.path.join(self.work_dir, "retccl_feats.npy"), retccl_array)

            progress.update(1)

        print(f"Features saved successfully to: {self.work_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Extract Image Features")
    
    parser.add_argument("--save_path", type=str, default="./data", help="Root data directory")
    parser.add_argument("--tissue_name", type=str, required=True, help="Tissue/Sample folder name")
    parser.add_argument("--img_path", type=str, required=True, help="Path to HE/WSI image")
    parser.add_argument("--spot_diameter_pixel", type=float, required=True, help="Spot diameter in pixels")
    parser.add_argument("--spot_diameter_um", type=float, default=55.0, help="Spot diameter in micrometers")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Directory containing RetCCL and HIPT checkpoint files")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device for feature extraction (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--retccl_device", type=str, default="cpu", help="Device for RetCCL model")
    parser.add_argument("--csv_filename", type=str, default="cell_segmented_coords.csv", help="CSV with cell coordinates")

    return parser.parse_args()


def main():
    args = parse_args()

    builder = FeatureConstruction(
        save_path=args.save_path,
        img_path=args.img_path,
        tissue_name=args.tissue_name,
        spot_diameter_pixel=args.spot_diameter_pixel,
        spot_diameter_um=args.spot_diameter_um,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        retccl_device=args.retccl_device,
        csv_filename=args.csv_filename,
    )
    builder.run_construct()


if __name__ == "__main__":
    main()