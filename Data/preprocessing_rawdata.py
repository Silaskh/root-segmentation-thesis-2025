### This script i used to preprocces all the raw data, it used the cutoff values (from the histogram files), and cuts off the background from the images. 
### Last edit: 10-02-2025
### By Agnes Lund Olsen and Silas Krongaard Hansen - based on notebook by Hans Martin Kjær
# Imports:
import qim3d
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import exposure

import os

#Defining functions

# Get the maximum and minimum values of the volume
def get_max_and_min(vol):
    return np.min(vol),np.max(vol)

# Generate histogram of the volume with specified cutoffs
def gen_histogram(vol,minVal,maxVal,min_cutoff,max_cutoff,name):
    output_dir_plots = "plots"
    os.makedirs(output_dir_plots, exist_ok=True)

    hist, bins = np.histogram(vol[::5,::5,::5], bins=int((maxVal-minVal)/10), density=True)
    binsCent = bins[0:-1] + np.divide(bins[1:]-bins[0:-1],2)
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.bar(binsCent,hist, width=(bins[1]-bins[0]))
    ax.plot([min_cutoff, min_cutoff],[0, max(hist)],'r')
    ax.plot([max_cutoff, max_cutoff],[0, max(hist)],'r')
    ax.set_ylim([0, 0.01])
    ax.set_title('Approximate histogram w/ new limits')

    image_path_1 = os.path.join(output_dir_plots, f"{name}_histogram1.png")
    image_path_2 = os.path.join(output_dir_plots, f"{name}_histogram2.png")

    if os.path.exists(image_path_1):
        fig.savefig(image_path_2)
    else:
        fig.savefig(image_path_1)

# Display and save slices of the volume
def display_and_save_slices(vol,n,name,min_val=None,max_val=None):
    slices=qim3d.viz.slices_grid(vol, num_slices=n, color_map='gray', value_min=min_val, value_max=max_val)
    output_dir_plots = "plots"
    os.makedirs(output_dir_plots, exist_ok=True)
    
    image_path_1 = os.path.join(output_dir_plots, f"{name}_slice1.png")
    image_path_2 = os.path.join(output_dir_plots, f"{name}_slice2.png")

    if os.path.exists(image_path_1):
        slices.savefig(image_path_2)
    else:
        slices.savefig(image_path_1) 

 
# Bit reduction of the volume based on specified cutoffs
def bit_reduction(vol,min_cutoff,max_cutoff):
    newMin = 0
    newMax = 1
    volCompress = np.float16(vol) 
    volCompress[volCompress < min_cutoff] = min_cutoff
    volCompress[volCompress > max_cutoff] = max_cutoff
    volCompress = (volCompress-min_cutoff) * ((newMax - newMin) / (max_cutoff-min_cutoff)) + newMin
    volCompress = np.uint8(volCompress * 255)
    return volCompress

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the volume
def apply_clahe(vol_uint8):
    vol_clahe = np.zeros_like(vol_uint8)
    for i in range(vol_uint8.shape[2]):
        vol_clahe[..., i] = exposure.equalize_adapthist(vol_uint8[..., i], clip_limit=0.03) * 255
    return vol_clahe.astype(np.uint8)

# Crop out a specific box from the volume and save it as a NIfTI file
def boxcropout(volCompress,header,name):
    output_dir_data = "processed_data"
    output_dir_plots = "plots"
    os.makedirs(output_dir_data, exist_ok=True)
    os.makedirs(output_dir_plots, exist_ok=True)

    boxSelect = np.array([[140,120,100],[1870,1900,1900]]) 
    sNo = 1000 

    fig, ax = plt.subplots(1,3,figsize=(24,8))
    ax[0].imshow(volCompress[...,sNo], cmap='gray')
    ax[0].plot(boxSelect[[0,1,1,0,0],1],boxSelect[[0,0,1,1,0],0], 'r', linewidth=5)
    ax[0].set_title('Slice XY ' + str(sNo))

    ax[1].imshow(np.squeeze(volCompress[:,sNo,:]), cmap='gray')
    ax[1].plot(boxSelect[[0,1,1,0,0],2],boxSelect[[0,0,1,1,0],0], 'r', linewidth=5)
    ax[1].set_title('Slice XZ ' + str(sNo))

    ax[2].imshow(np.squeeze(volCompress[sNo,:,:]), cmap='gray')
    ax[2].plot(boxSelect[[0,1,1,0,0],2],boxSelect[[0,0,1,1,0],1], 'r', linewidth=5)
    ax[2].set_title('Slice YZ ' + str(sNo))

    image_path_1 = os.path.join(output_dir_plots, f"{name}_boxselection_1.png")
    image_path_2 = os.path.join(output_dir_plots, f"{name}_boxselection_2.png")

    if os.path.exists(image_path_1):
        fig.savefig(image_path_2)
    else:
        fig.savefig(image_path_1)

    volCrop = volCompress[boxSelect[0,0]:boxSelect[1,0] , boxSelect[0,1]:boxSelect[1,1] , boxSelect[0,2]:boxSelect[1,2]]

    exportFile1 = name+"_processed1.nii"
    exportFile2 = name+"_processed2.nii"
    pixSz = header['volume1']['file1']['volumeprimitive1']['geometry']['resolution'].split(' ')
    imgSpace = np.array([ float(pixSz[0]), float(pixSz[1]), float(pixSz[2]) ]) # mm

    origin = np.array([0,0,0])
    affine = np.zeros((4,4))
    affine[0:3,3] = origin
    affine[0,0] = imgSpace[0]
    affine[1,1] = imgSpace[1]
    affine[2,2] = imgSpace[2]

    volNii = nib.Nifti1Image(volCrop, affine)
    
    if os.path.exists(output_dir_data + exportFile1):
        nib.save(volNii, output_dir_data + exportFile2)
    else:
        nib.save(volNii, output_dir_data + exportFile1)
    
    print(f"Processed files saved in '{output_dir_data}'")




PREFIX =  "/zhome/4f/d/187167/"

FOLDER_LIST= [("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_11 [2024-11-05 12.59.48]/week1_11_bottom_recon/","week1_11.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_11 [2024-11-05 13.23.38]/week1_11_top_recon/","week1_11.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_12 [2024-11-05 12.01.14]/week1_12_bottom_recon/","week1_12.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_12 [2024-11-05 12.25.00]/week1_12_top_recon/","week1_12.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_13 [2024-11-05 14.45.55]/week1_13_bottom_recon/","week1_13.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_13 [2024-11-05 15.09.46]/week1_13_top_recon/","week1_13.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_14 [2024-11-05 15.39.04]/week1_14_bottom_recon/","week1_14.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_14 [2024-11-05 16.02.56]/week1_14_top_recon/","week1_14.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_15 [2024-11-05 11.04.45]/week1_15_bottom_recon/","week1_15.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_15 [2024-11-05 11.28.38]/week1_15_top_recon/","week1_15.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_16 [2024-11-05 13.53.05]/week1_16_bottom_recon/","week1_16.vgi"),
              ("QIM/projects/2024_DANFIX_147_RAPID-P/raw_data_3DIM/week1/week1_16 [2024-11-05 14.16.56]/week1_16_top_recon/","week1_16.vgi")]

for folder, file in FOLDER_LIST:
    vol,header =qim3d.io.load(PREFIX+folder+file,return_metadata=True) # load
    minVal,maxVal=get_max_and_min(vol)
    cutoff_min_val,cutoff_max_val=-10,125
    name = file.split(".")[0]
    gen_histogram(vol,minVal,maxVal,cutoff_min_val,cutoff_max_val,name)
    display_and_save_slices(vol,10,name,min_val=cutoff_min_val,max_val=cutoff_max_val)#Display slices after cutoff
    vol_reduced = bit_reduction(vol,cutoff_min_val,cutoff_max_val)
    vol_reduced = apply_clahe(vol_reduced)
    display_and_save_slices(vol_reduced, 10, name + "_clahe")
    boxcropout(vol_reduced,header,name)

    print("min,max cutoff values:", cutoff_min_val,cutoff_max_val)
