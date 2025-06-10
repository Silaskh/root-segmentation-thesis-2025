print("Opened file")
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import qim3d
import torch
from pathlib import Path


def load_single_segmentation(user_id, segment_path=r"QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Preprocessing/processed_data/Segmentations/new_seg_week2_30_1_para500.nii.gz"): #"QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Model/Legacy_Predictions/small_result.nii.gz"):
    segment_file = user_id + segment_path
    root_labels, label_header = qim3d.io.load(segment_file, return_metadata=True)
    root_labels = torch.from_numpy(root_labels).float()
    return root_labels, label_header



ID = str(Path.home()) + "/"
NAME = "small_crop"
segment_folder = "/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Model/local_test/Local_testing/200_1/"
segment_file = "overfit_patch_model.pth"
segmentation_data , header = load_single_segmentation(ID,segment_path = segment_folder + segment_file)


print("\n", segmentation_data.shape, "\n")

print("Unique values:", np.unique(segmentation_data), "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_data = segmentation_data.to(device)
tensor_data = (tensor_data >= 0.5).to(torch.uint8) 
tensor_data[tensor_data == 2] = 0 
projection0 = tensor_data.sum(dim=0)

projection1 = tensor_data.sum(dim=1)  
projection2 = tensor_data.sum(dim=2)    


projection_cpu0 = projection0.cpu().numpy()
projection_cpu1 = projection1.cpu().numpy()
projection_cpu2 = projection2.cpu().numpy()

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.imshow(projection_cpu0, cmap='hot', interpolation='nearest')
plt.title("2D Projection Heatmap - Axis 0")
plt.colorbar(label="Summed Value")
plt.xlabel("X axis")
plt.ylabel("Y axis")

saved = os.path.join("plots", NAME+"_axis0.png")
plt.savefig(saved)
plt.close()

print("Saved heatmap image at:", saved)

plt.figure(figsize=(8, 6))
plt.imshow(projection_cpu1, cmap='hot', interpolation='nearest')
plt.title("2D Projection Heatmap - Axis 1")
plt.colorbar(label="Summed Value")
plt.xlabel("X axis")
plt.ylabel("Y axis")

saved = os.path.join("plots", NAME+"_axis1.png")
plt.savefig(saved)
plt.close()

print("Saved heatmap image at:", saved)

plt.figure(figsize=(8, 6))
plt.imshow(projection_cpu2, cmap='hot', interpolation='nearest')
plt.title("2D Projection Heatmap - Axis 2")
plt.colorbar(label="Summed Value")
plt.xlabel("X axis")
plt.ylabel("Y axis")

saved = os.path.join("plots", NAME+"_axis2.png")
plt.savefig(saved)
plt.close()

print("Saved heatmap image at:", saved)
