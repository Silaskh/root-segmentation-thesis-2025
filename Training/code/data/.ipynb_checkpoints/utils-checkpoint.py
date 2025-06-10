# file_utils.py

import os
from typing import List
import nibabel as nib
import numpy as np
from pathlib import Path


def extract_id(file_path: str) -> int:
    import re
    basename = os.path.basename(file_path)
    match = re.findall(r'\d+', basename)
    if not match:
        raise ValueError(f"Could not extract ID from filename: {basename}")
    return int(match[-1])
    
def list_files_in_directory(directory_path: str, absolute: bool = True) -> List[str]:
    if not os.path.isdir(directory_path):
        raise ValueError(f"The path '{directory_path}' is not a valid directory.")

    file_list = [
        os.path.join(directory_path, f) if absolute else f
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    
    return file_list

def load_nifti_as_numpy(file_path: str) -> np.ndarray:
    if not file_path.endswith(".nii") and not file_path.endswith(".nii.gz"):
        raise ValueError(f"File '{file_path}' is not a NIfTI file.")

    img = nib.load(file_path)
    data = img.get_fdata()  # use get_fdata() to get data as float64
    return np.array(data)   # explicitly convert to ensure np.ndarray type
    

def compute_array_sum(array: np.ndarray) -> float:
    return float(np.sum(array))


def make_path_portable(path: str) -> str:
    """
    Replaces the user's home path with '~' to make paths portable across users.

    Args:
        path (str): Full absolute path.

    Returns:
        str: Path with home directory replaced by '~'.
    """
    return str(path).replace(str(Path.home()), "~")

def foo():
    print("bar")