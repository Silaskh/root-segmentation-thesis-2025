from pathlib import Path
from monai.data import PersistentDataset, DataLoader
from typing import List, Dict, Optional, Callable
import os
from code.data.utils import list_files_in_directory,extract_id

#This module handles data loading for MONAI-compatible datasets.


#Replace tilde (~) in paths with the user's home directory.
def expand_user_paths(data_list: List[Dict[str, str]], in_place: bool = False) -> List[Dict[str, str]]:
    if in_place:
        for item in data_list:
            item['image'] = str(Path(item['image']).expanduser())
            item['label'] = str(Path(item['label']).expanduser())
        return data_list
    else:
        return [
            {
                **item,
                'image': str(Path(item['image']).expanduser()),
                'label': str(Path(item['label']).expanduser())
            }
            for item in data_list
        ]
    
#Builds a dataset list from image and label folders, pairing files by extracted numeric IDs.
def build_dataset_list_from_folders(
    image_dir: str,
    label_dir: str,
    sort: bool = True
) -> List[Dict]:
    home = str(Path.home())
    
    image_files = list_files_in_directory(image_dir, absolute=True)
    label_files = list_files_in_directory(label_dir, absolute=True)

    if len(image_files) != len(label_files):
        raise ValueError(
            f"Image/label count mismatch: {len(image_files)} images, {len(label_files)} labels."
        )

    if sort:
        image_files = sorted(image_files, key=extract_id)
        label_files = sorted(label_files, key=extract_id)

    dataset = []
    for idx, (img, lbl) in enumerate(zip(image_files, label_files)):
        dataset.append({
            "image": img.replace(home, "~"),
            "label": lbl.replace(home, "~"),
            "id": idx
        })

    return dataset

# Creates a MONAI DataLoader using PersistentDataset for efficient data loading.
def get_data_loader(
    data_list: List[Dict],
    transforms: Callable,
    cache_dir: str = "./persistent_cache",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    
    dataset = PersistentDataset(
        data=data_list,
        transform=transforms,
        cache_dir=cache_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader
