import os 
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
from cellpose import models
from squidpy.im import ImageContainer
import squidpy as sq
import skimage.measure

class NucleiSegment(object):
    def __init__(self, save_path, img_path, tissue_name, patch_size, flow_threshold=0.4, diameter=None):
        """Initialize NucleiSegment class."""
        if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
            raise ValueError("patch_size must be a tuple of (height, width)")
        if patch_size[0] <= 0 or patch_size[1] <= 0:
            raise ValueError("patch_size values must be positive integers")

        if flow_threshold <= 0:
            raise ValueError("flow_threshold must be positive")
        if diameter is not None and diameter <= 0:
            raise ValueError("diameter must be positive or None")
            
        self.save_path = save_path
        self.tissue_name = tissue_name
        
        self.work_dir = os.path.join(save_path, self.tissue_name)
        self.crop_outdir = os.path.join(self.work_dir, 'crop_image')

        os.makedirs(self.crop_outdir, exist_ok=True)

        self.image = cv2.imread(img_path)
        if self.image is None:
            raise ValueError(f"Image at {img_path} could not be loaded.")
            
        self.patch_size = patch_size
        self.flow_threshold = flow_threshold
        self.diameter = diameter
        self.model = models.Cellpose(model_type='nuclei') 

    def _cellpose_segment(self, img, min_size=15, channel_cellpose=2):
        """Helper function for Cellpose segmentation."""
        res, _, _, _ = self.model.eval(
            img,
            channels=[channel_cellpose, 0],
            diameter=self.diameter,
            min_size=min_size,
            invert=True,
            flow_threshold=self.flow_threshold
        )
        return res

    def split_img(self):
        """Split image into patches and calculate split coordinates."""
        height, width = self.image.shape[:2]

        self.x_list = list(range(0, width, self.patch_size[1]))
        if self.x_list[-1] != width:
            self.x_list.append(width)

        self.y_list = list(range(0, height, self.patch_size[0]))
        if self.y_list[-1] != height:
            self.y_list.append(height)

    def compress_img_CV(self, compress_rate=0.5):
        height, width = self.image.shape[:2]
        img_resize = cv2.resize(
            self.image,
            (int(width * compress_rate), int(height * compress_rate)),
            interpolation=cv2.INTER_AREA)
        return img_resize

    def run_segment(self):
        """Run segmentation on image patches and save results."""
        if not hasattr(self, 'x_list') or not hasattr(self, 'y_list'):
            raise RuntimeError("Please call split_img() before run_segment()")

        centr = pd.DataFrame()
        
        for i in range(len(self.y_list)-1):  
            for j in range(len(self.x_list)-1):  
                y1, y2 = self.y_list[i], self.y_list[i+1]
                x1, x2 = self.x_list[j], self.x_list[j+1]
                
                patch = self.image[y1:y2, x1:x2]
                
                crop = ImageContainer()
                crop.add_img(patch, layer="image")
                sq.im.segment(img=crop, layer="image", channel=None, method=self._cellpose_segment)
                
                library_id = crop._get_library_id(None)
                label_arr = crop["segmented_custom"].sel(z=library_id)
                label_arr_0 = label_arr[..., 0].values
                
                tmp_features = skimage.measure.regionprops_table(
                    label_arr_0, 
                    properties=["label", "centroid"]
                )
                tmp_features_df = pd.DataFrame.from_dict(tmp_features)
                
                tmp_features_df["centroid-0"] += y1  
                tmp_features_df["centroid-1"] += x1  
                
                centr = pd.concat([centr, tmp_features_df], ignore_index=True)
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 20))
                crop.show("image", channel=None, ax=axes[0])
                axes[0].set_title("H&E")
                crop.show("segmented_custom", cmap="jet", interpolation="none", ax=axes[1])
                axes[1].set_title("Cellpose segmentation")
                
                cell_number = len(tmp_features["label"])
                filename = f"crop_{i}_{j}_{cell_number}.pdf"
                plt.savefig(os.path.join(self.crop_outdir, filename))
                plt.close()  

        out_df = centr[["centroid-1", "centroid-0"]].copy().rename(columns={'centroid-1': 'pixel_x', 'centroid-0': 'pixel_y'})
        csv_out_path = os.path.join(self.work_dir, "cell_segmented_coords.csv")
        out_df.to_csv(csv_out_path, index=False)
        print(f"Coordinates successfully saved to {csv_out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Nuclei Segmentation using Cellpose")
    
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--tissue_name", type=str, default="tissue_name", help="Name of the tissue sample folder")
    parser.add_argument("--img_name", type=str, default="tissue_section.png", help="Image file name inside data_dir/tissue_name/")
    parser.add_argument("--img_path", type=str, default=None, help="Direct full path to the image file (optional)")
    
    parser.add_argument("--patch_size", type=int, nargs=2, default=[5000, 5000], help="Patch size as height width")
    parser.add_argument("--flow_threshold", type=float, default=0.4, help="Cellpose parameter for boundary detection")
    parser.add_argument("--diameter", type=int, default=10, help="Expected diameter of nuclei for Cellpose")
    
    return parser.parse_args()


    

if __name__ == "__main__":
    args = parse_args()
    
    if args.img_path is not None:
        img_path = args.img_path
    else:
        img_path = os.path.join(args.data_dir, args.tissue_name, args.img_name)
    
    save_path = args.data_dir
    sample_dir = os.path.join(save_path, args.tissue_name)
    
    print(f"Loading image from: {img_path}")
    print(f"All outputs will be saved inside: {sample_dir}")
    
    nuclei_segmentation = NucleiSegment(
        save_path=save_path,
        img_path=img_path,
        tissue_name=args.tissue_name,
        patch_size=tuple(args.patch_size),
        flow_threshold=args.flow_threshold,  
        diameter=args.diameter
    )

    nuclei_segmentation.split_img()
    nuclei_segmentation.run_segment()
    print("Segmentation completed successfully.")
    
    