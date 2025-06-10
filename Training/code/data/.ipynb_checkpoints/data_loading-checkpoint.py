from pathlib import Path
from monai.data import PersistentDataset, DataLoader
from typing import List, Dict, Optional, Callable
import os
from code.data.utils import list_files_in_directory,extract_id

def expand_user_paths(data_list: List[Dict[str, str]], in_place: bool = False) -> List[Dict[str, str]]:
    """
    Expands '~' in 'image' and 'label' paths to the full user path.

    Args:
        data_list (List[Dict[str, str]]): List of dicts with 'image' and 'label' keys.
        in_place (bool): If True, modifies the original list. If False, returns a new list.

    Returns:
        List[Dict[str, str]]: List with expanded paths (if in_place=False).
    """
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
def build_dataset_list_from_folders(
    image_dir: str,
    label_dir: str,
    sort: bool = True
) -> List[Dict]:
    """
    Constructs a MONAI-compatible dataset list by pairing image and label files.

    Args:
        image_dir (str): Folder containing image (input) files.
        label_dir (str): Folder containing label (segmentation) files.
        sort (bool): If True, sorts by extracted numeric ID before pairing.

    Returns:
        List[Dict]: List of {'image': str, 'label': str, 'id': int}
    """
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

def get_data_loader(
    data_list: List[Dict],
    transforms: Callable,
    cache_dir: str = "./persistent_cache",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Creates and returns a MONAI DataLoader using PersistentDataset.

    Args:
        data_list (List[Dict]): List of dictionaries with 'image' and 'label' keys.
        transforms (Callable): MONAI transform pipeline to apply.
        cache_dir (str): Directory for persistent caching.
        batch_size (int): Batch size for loading data.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to use pinned memory (for CUDA).

    Returns:
        DataLoader: Configured MONAI DataLoader.
    """
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
