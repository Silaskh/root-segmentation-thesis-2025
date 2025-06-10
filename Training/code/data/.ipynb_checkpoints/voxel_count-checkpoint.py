import os
import json
from typing import List, Dict
from pathlib import Path
from code.data.utils import (
    list_files_in_directory,
    load_nifti_as_numpy,
    compute_array_sum,
    extract_id
)

def make_path_portable(path: str) -> str:
    return str(path).replace(str(Path.home()), "~")



def filter_and_report_dataset(
    volume_dir: str,
    segmentation_dir: str,
    save_to_txt: bool = False,
    output_path: str = "filtered_dataset.txt"
) -> List[Dict]:
    """
    Filters out segmentations with zero sum and returns a list of usable samples
    with remapped contiguous IDs starting from 0.

    Args:
        volume_dir (str): Path to volume files.
        segmentation_dir (str): Path to segmentation files.
        save_to_txt (bool): Whether to save result to a .txt file.
        output_path (str): Output path for the result file.

    Returns:
        List[Dict]: [{'image': path, 'label': path, 'id': int}, ...]
    """
    volume_files = list_files_in_directory(volume_dir, absolute=True)
    segmentation_files = list_files_in_directory(segmentation_dir, absolute=True)

    id_to_volume = {extract_id(f): f for f in volume_files}
    id_to_seg = {extract_id(f): f for f in segmentation_files}

    result = []
    sorted_seg_ids = sorted(id_to_seg.keys())
    new_id = 0

    for seg_id in sorted_seg_ids:
        seg_path = id_to_seg[seg_id]

        if seg_id not in id_to_volume:
            print(f"[WARN] Segmentation {seg_id} has no matching volume. Skipping.")
            continue

        array = load_nifti_as_numpy(seg_path)
        seg_sum = compute_array_sum(array)

        if seg_sum == 0:
            print(f"[ID {seg_id}] Skipped (sum = 0)")
            continue

        print(f"[ID {seg_id}] Segmentation sum: {seg_sum}")
        result.append({
            'image': make_path_portable(id_to_volume[seg_id]),
            'label': make_path_portable(seg_path),
            'id': new_id
        })
        new_id += 1

    print("\nFiltered dataset:")
    print(result)

    if save_to_txt:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved filtered dataset to {output_path}")

    return result

if __name__ == "__main__":
    # Change these paths to the actual directories
    volume_dir = "/zhome/4f/d/187167/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/"
    segmentation_dir = "/zhome/4f/d/187167/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/"
    save_output = True  # Set to False to skip saving

    filter_and_report_dataset(volume_dir, segmentation_dir, save_to_txt=save_output)
