

import os
from typing import List
import nibabel as nib
import numpy as np
from pathlib import Path

# Module containing utility functions for data processing


#Extracts an integer ID from a file path, assuming the ID is embedded in the filename.
def extract_id(file_path: str) -> int:
    import re
    basename = os.path.basename(file_path)
    match = re.findall(r'\d+', basename)
    if not match:
        raise ValueError(f"Could not extract ID from filename: {basename}")
    return int(match[-1])
    
# Lists all files in a directory, returning their paths as absolute or relative based on the 'absolute' parameter.
def list_files_in_directory(directory_path: str, absolute: bool = True) -> List[str]:
    if not os.path.isdir(directory_path):
        raise ValueError(f"The path '{directory_path}' is not a valid directory.")

    file_list = [
        os.path.join(directory_path, f) if absolute else f
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    
    return file_list

# Loads a NIfTI file and returns its data as a NumPy array.
def load_nifti_as_numpy(file_path: str) -> np.ndarray:
    if not file_path.endswith(".nii") and not file_path.endswith(".nii.gz"):
        raise ValueError(f"File '{file_path}' is not a NIfTI file.")

    img = nib.load(file_path)
    data = img.get_fdata()  
    return np.array(data)   
    

# Computes the sum of all elements in a NumPy array and returns it as a float.
def compute_array_sum(array: np.ndarray) -> float:
    return float(np.sum(array))

# Converts a path to a portable format by replacing the user's home directory with "~".
def make_path_portable(path: str) -> str:
    return str(path).replace(str(Path.home()), "~")

