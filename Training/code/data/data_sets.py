from code.data.data_loading import build_dataset_list_from_folders
import os


#This module contains the datasets used in the project.
#They are not designed to be reused but a structure for how further datasets can be created.
def get_data_set(data):
    if data == None:
        return None
    dict = {"week3_30_2_400_all": og_res_week3_30_2_400_all(),
            "week3_30_2_400_only_root": og_res_week3_30_2_400_only_root(),
            "week2_30_2_400_all":  down_1_res_week2_30_2_400_all(),
            "week2_30_2_400_only_root": down_1_res_week2_30_2_400_only_root(),
            "week2_30_2_400_mask": week2_30_2_400_mask(),
            "week2_30_bottom_DOWN":week2_30_bottom_DOWN(),
            "week2_30_top_DOWN": week2_30_top_DOWN(),
            "predict_third": predict_third(),
            "w3_DOWN":w3_DOWN(),
            "w2_30_2_full": w2_30_2_full(),
            "down_1_res_week2_30_top_400_all": down_1_res_week2_30_top_400_all(),
            "w2_30_2_mini_patch":w2_30_2_mini_patch()
           
              }
    return dict[data]


def w2_30_2_mini_patch():
    return [{"image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/Patches/2_30_patch_image.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/Patches/2_30_patch_label.nii.gz",
    "id":0}]
    
    

def down_1_res_week2_30_top_400_all():
    seg_path = "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_top_1/"
    pro_path = "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_top/"
    return build_dataset_list_from_folders(os.path.expanduser(pro_path), os.path.expanduser(seg_path))


def w2_30_2_full():
    return [{"image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/Week2/week2_30_bottom.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/Week2/week2_30_bottom_root-class.nii.gz",
    "id":0}]
def w3_DOWN():
    return [
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week3_30_bottom_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week3_30_bottom_segmentation_sd.nii.gz",
    "id": 0
  },{
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week3_30_top_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week3_30_top_segmentation_sd.nii.gz",
    "id": 1
  }]

def predict_third():
    return [
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_bottom_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_bottom_segmentation_sd.nii.gz",
    "id": 0
  },{
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_top_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_top_segmentation_sd.nii.gz",
    "id": 1
  },{
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_31_bottom_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_31_bottom_segmentation_sd.nii.gz",
    "id": 2
  },{
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_31_top_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_31_top_segmentation_sd.nii.gz",
    "id": 3
  }]
def week2_30_bottom_DOWN():
    return [
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_bottom_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_bottom_segmentation_sd.nii.gz",
    "id": 0
  } ]
  
def week2_30_top_DOWN():
    return [
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_top_sd.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/NonCropped/super_down/week2_30_top_segmentation_sd.nii.gz",
    "id": 0
  } ]

def week2_30_2_400_mask():
    return [
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_1.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_1.nii.gz",
    "id": 0
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_2.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_2.nii.gz",
    "id": 1
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_3.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_3.nii.gz",
    "id": 2
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_5.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_5.nii.gz",
    "id": 3
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_9.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_9.nii.gz",
    "id": 4
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_10.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_10.nii.gz",
    "id": 5
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_11.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_11.nii.gz",
    "id": 6
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_12.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_12.nii.gz",
    "id": 7
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_13.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_13.nii.gz",
    "id": 8
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_14.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_14.nii.gz",
    "id": 9
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_15.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_15.nii.gz",
    "id": 10
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_16.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_16.nii.gz",
    "id": 11
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_17.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_17.nii.gz",
    "id": 12
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_18.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_18.nii.gz",
    "id": 13
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_19.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_19.nii.gz",
    "id": 14
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_21.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_21.nii.gz",
    "id": 15
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_22.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_22.nii.gz",
    "id": 16
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_23.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1_mask/week2_30_2_1_mask_400_400_400_0.1_segmentation_cropped_23.nii.gz",
    "id": 17
  }
]

def og_res_week3_30_2_400_all():
            return build_dataset_list_from_folders(os.path.expanduser("~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/"), os.path.expanduser("~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/"))
def down_1_res_week2_30_2_400_all():
        return build_dataset_list_from_folders(os.path.expanduser("~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/"), os.path.expanduser("~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/"))

def down_1_res_week2_30_2_400_only_root():
    return [
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_1.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_1.nii.gz",
    "id": 0
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_3.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_3.nii.gz",
    "id": 1
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_9.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_9.nii.gz",
    "id": 2
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_10.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_10.nii.gz",
    "id": 3
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_12.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_12.nii.gz",
    "id": 4
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_13.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_13.nii.gz",
    "id": 5
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_14.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_14.nii.gz",
    "id": 6
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_18.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_18.nii.gz",
    "id": 7
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_19.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_19.nii.gz",
    "id": 8
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_21.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_21.nii.gz",
    "id": 9
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_22.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_22.nii.gz",
    "id": 10
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Week2/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_data_cropped_23.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Downsampled/Cropped/Segmentations/Week2_30_2_1/week2_30_2_1_400_400_400_0.1_segmentation_cropped_23.nii.gz",
    "id": 11
  }
]
    
    
def og_res_week3_30_2_400_only_root(): 
    return [ {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_14.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_14.nii.gz",
    "id": 0
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_15.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_15.nii.gz",
    "id": 1
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_50.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_50.nii.gz",
    "id": 2
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_51.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_51.nii.gz",
    "id": 3
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_85.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_85.nii.gz",
    "id": 4
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_86.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_86.nii.gz",
    "id": 5
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_87.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_87.nii.gz",
    "id": 6
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_91.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_91.nii.gz",
    "id": 7
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_92.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_92.nii.gz",
    "id": 8
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_121.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_121.nii.gz",
    "id": 9
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_122.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_122.nii.gz",
    "id": 10
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_123.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_123.nii.gz",
    "id": 11
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_127.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_127.nii.gz",
    "id": 12
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_128.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_128.nii.gz",
    "id": 13
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_157.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_157.nii.gz",
    "id": 14
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_158.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_158.nii.gz",
    "id": 15
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_159.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_159.nii.gz",
    "id": 16
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_164.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_164.nii.gz",
    "id": 17
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_193.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_193.nii.gz",
    "id": 18
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_194.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_194.nii.gz",
    "id": 19
  },
  {
    "image": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Week3/30_2_400/week3_30_2_400_400_400_0.2_data_cropped_200.nii.gz",
    "label": "~/QIM/projects/2024_DANFIX_147_RAPID-P/analysis/BachelorProject/Data/Original_resolution/Cropped/Segmentations/Week3_30_2/week3_30_2_400_400_400_0.2_segmentation_cropped_200.nii.gz",
    "id": 20
  }
]
