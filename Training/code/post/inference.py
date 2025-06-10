import torch
import nibabel as nib
import numpy as np
import os
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity
import os
import matplotlib.pyplot as plt

#Module containing functions for running inference and generating heatmaps from 3D NIfTI images

#Inference function using sliding window approach
def run_sliding_window_inference(
    model,
    image_path,
    output_path,
    window_size=(416, 416, 416),
    overlap=0.25,
    progress = True,
):
    
    normalize = NormalizeIntensity()
    model.eval()
    model.cuda() 

    img_nib = nib.load(image_path)
    affine = img_nib.affine
    image_data = img_nib.get_fdata().astype(np.float32)
    input_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0).cpu() 
    
    input_tensor = normalize(input_tensor)
    input_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=window_size,
            sw_batch_size=1,
            predictor=lambda x: torch.sigmoid(model(x.cuda())).cpu(),
            overlap=overlap,
            mode="gaussian",
            device="cpu",  
            progress = progress
        )

    prediction = output.squeeze().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(prediction.astype(np.float32), affine), output_path)
    print(f"Saved sliding window prediction to: {output_path}")

#Function to generate projection heatmaps from a 3D NIfTI file
def generate_projection_heatmaps(
    nifti_path: str,
    output_dir: str,
    name: str = "heatmap",
    root_label: int = 1
):
    
    os.makedirs(output_dir, exist_ok=True)

    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    if data.ndim != 3:
        raise ValueError("Expected 3D volume.")

    data = (data >= 0.5).astype(np.uint8) 
    data[data != root_label] = 0  

    tensor = torch.from_numpy(data)

    for axis in range(3):
        projection = tensor.sum(dim=axis).cpu().numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(projection, cmap='hot', interpolation='nearest')
        plt.title(f"Projection Heatmap - Axis {axis}")
        plt.colorbar(label="Voxel Count")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")

        out_path = os.path.join(output_dir, f"{name}_axis{axis}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved heatmap: {out_path}")
        
if __name__ == "__main__":
    # Main entry point for running inference and generating heatmaps
    import argparse
    from monai.networks.nets import UNet

    parser = argparse.ArgumentParser(description="Run inference and heatmap generation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (.nii.gz)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save predicted output (.nii.gz)")
    parser.add_argument("--heatmap_dir", type=str, required=True, help="Directory to save heatmap PNGs")
    parser.add_argument("--window_size", type=int, nargs=3, default=[416, 416, 416], help="Sliding window size")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap fraction")

    args = parser.parse_args()

   
    unet_base = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    )


    checkpoint = torch.load(args.model_path, map_location="cuda")
    unet_base.load_state_dict(checkpoint)
    print(f"Loaded model from {args.model_path}")


    run_sliding_window_inference(
        model=unet_base,
        image_path=args.image_path,
        output_path=args.output_path,
        window_size=tuple(args.window_size),
        overlap=args.overlap
    )


    generate_projection_heatmaps(
        nifti_path=args.output_path,
        output_dir=args.heatmap_dir,
        name=os.path.splitext(os.path.basename(args.output_path))[0]
    )

