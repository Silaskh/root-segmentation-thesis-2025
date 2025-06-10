
# Libraries 
import qim3d
import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math 
import matplotlib.cm as cm
import matplotlib.colors as colors
import gc  
from skimage.segmentation import find_boundaries

print("Running full plot analysis: Depth level 3 making probability, count and density plots, as well as a report file")

#Sample specifications: 

SAMPLE = "Sample_30_Week_2"
SAMPLE_PATH = "QIM/projects/2024_DANFIX_147_RAPID-P/analysis/sample30/s30_joint/week2-joint-root-class.nii.gz"

#Image-specific variables
center_voxel = (388,223,223)
voxel_cylinder_edges = [(15,223),(430,223),(223,25),(223,440)] 
# Plot variables
NUM_CIRCLES = 7
NUM_ANGLES = 7
DEPTH_SEGMENTS=3
TYPE = 2  # 0 Circle, 1 Angle, 2 Combined
ALL_LABELS = np.arange(NUM_CIRCLES * NUM_ANGLES)

type_dict = {0:"circle", 1:"angle", 2:"combined"}
count_dict = {0:"count", 1:"probability", 2:"density"}
COLORMAP = "Greens"
BORDER_COLOR= [0, 0, 0]  
BACKGROUND_COLOR = [255, 255, 255]  # Draw white background 
#BACKGROUND_COLOR = [220, 220, 220]  # Draw light gray background



PLOT_TITLE = False # TRUE if Main titel is Root Voxel Probability Distribution, FALSE if main title is depth segment
PERCENT_SHOW = True # TRUE if you want to show the percent of root voxel in the depth slice  root. 

# For the report file
report_filename = f"report_{SAMPLE}.txt"
summary_lines = []  

summary_lines.append(f"Plot settings:")
summary_lines.append(f"  - Number of circles: {NUM_CIRCLES}")
summary_lines.append(f"  - Number of angular bins: {NUM_ANGLES}")
summary_lines.append(f"  - Number of depth segments: {DEPTH_SEGMENTS}")
summary_lines.append("")  
summary_lines.append(f"Image-specific points:")
summary_lines.append(f"  - Center voxel: {center_voxel}")
summary_lines.append(f"  - Voxel cylinder edges: {voxel_cylinder_edges}")



id = str(Path.home()) + "/"
seg_vol, seg_header = qim3d.io.load(id + SAMPLE_PATH , return_metadata=True)


seg_vol[seg_vol == 2] = 0 
seg_vol = seg_vol > 0
print("Volume loaded and binarized")



def create_mesh_grid(volume, center_voxel, step_voxel=(1,1,1)):
    shape = np.array(volume.shape)
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    Y, Z = np.meshgrid(y, z, indexing='ij')
    Y = (Y - center_voxel[1] / step_voxel[1])    
    Z = (Z - center_voxel[2] / step_voxel[2])  
    return Y, Z

def voxel_to_coord(voxel_pos, Y, Z):
    return (Y[voxel_pos], Z[voxel_pos])

def list_voxel_to_coord(voxel_list, Y, Z):
    coordinates = []
    for i in voxel_list:
        coordinates.append(voxel_to_coord(i, Y, Z))
    return coordinates

def make_large_radius(cylinder_edges, center_voxel, circles_amount):
    mean_radius = 0
    for i in range(len(cylinder_edges)):
        mean_radius += np.sqrt(cylinder_edges[i][0]**2 + cylinder_edges[i][1]**2)
    mean_radius /= len(cylinder_edges)
    radius_length = mean_radius / circles_amount
    radii = []
    radius = radius_length
    for i in range(circles_amount):
        radii.append(radius)
        radius += radius_length
    return radii, mean_radius

def circle_levels_divider(radius_list, Y, Z):
    distance_center = np.sqrt(Y**2 + Z**2)
    labels = np.searchsorted(radius_list, distance_center)
    labels[distance_center > radius_list[-1]] = -1
    return labels


def angular_slicing(num):
    angle = 360 / num
    angles = []
    for i in range(num + 1):
        angles.append(angle * i)
    return angles

def angle_levels_divider(angles_list, Y, Z):
    angle_radians = np.arctan2(Z, Y)
    angle_deg = np.degrees(angle_radians)
    angle_deg = (angle_deg + 360) % 360  

    labels = np.searchsorted(angles_list, angle_deg, side='right') - 1
    labels = np.clip(labels, 0, len(angles_list) - 2)
    
    
    return labels


def combine_labels(label_circle, label_angle, NUM_ANGLES):
    return label_circle * NUM_ANGLES + label_angle

def check_outside_cylinder(Y, Z, radius):
    return np.sqrt(Y**2 + Z**2) > radius

def Area(NUM_ANGLES, NUM_CIRCLES, angles, radius_list):
    areas = []
    angle_step = angles_list[1]
    for i in range(NUM_ANGLES):
        A = (angles[1]/2) * (radius_list[0]**2) 
        areas.append(A)
    for k in range(1, NUM_CIRCLES):
        for i in range(NUM_ANGLES):
            A = (angle_step /2) * (radius_list[k]**2 - radius_list[k-1]**2)
            areas.append(A)
    return areas


def count_normalization(volume, labels, unique_labels):
    print("total count across bins:", np.sum( np.array([np.sum(volume[labels == label]) for label in unique_labels])))
    return np.array([np.sum(volume[labels == label]) for label in unique_labels])

def probability_normalization(volume, labels, unique_labels, global_total=None):
    counts = count_normalization(volume, labels, unique_labels)
    total = global_total if global_total is not None else np.sum(counts)
    return counts / total


def voxel_count(labels, unique_labels):
    return np.array([np.sum(labels == label) for label in unique_labels])

def return_values(count_type, volume, labels, unique_labels, circle_labels, areas, global_total=None):
    if count_type == 0:
        return count_normalization(volume, labels, unique_labels)
    elif count_type == 1:
        return probability_normalization(volume, labels, unique_labels, global_total=global_total)
    elif count_type == 2:
        valid_mask = circle_labels != -1
        valid_labels = labels[valid_mask]
        valid_volume = volume[valid_mask]

        valid_unique_labels = ALL_LABELS 

        counts = probability_normalization(valid_volume, valid_labels, valid_unique_labels, global_total=global_total)

        areas = np.array(areas)
        counts_divided = counts / (areas + 1e-8)
        return counts_divided
    


def plot_cylinder_sectioning_single(
    labels,
    circle_labels,
    volume,
    count_type,
    Area,
    colormap="inferno",
    save=True,
    plot_name="cylinder_plot",
    plot_format=".png",
    plot_path="plots/",
    global_total=None,
    local_total=None, 
    max_val=None 
):
    labels = np.copy(labels)
    labels[labels >= len(Area)] = -1  

    unique_labels = ALL_LABELS 

    intensity_sums = return_values(count_type, volume, labels, unique_labels, circle_labels, Area, global_total=global_total)
    
    norm_sums = intensity_sums 
    cmap = plt.get_cmap(colormap)
    rgb_colors = (np.array([cmap(val/max_val)[:3] for val in norm_sums]) * 255).astype(np.uint8)

    output_image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i, label in enumerate(unique_labels):
        output_image[labels == label] = rgb_colors[i]

    valid_labels = np.copy(labels)
    valid_labels[circle_labels < 0] = -1  

    boundaries = find_boundaries(valid_labels, mode='outer')
    output_image[boundaries] = BORDER_COLOR

    output_image[circle_labels < 0] = BACKGROUND_COLOR  

    fig, ax = plt.subplots()
    ax.imshow(output_image.astype(np.uint8))
    label_dict = {0: "Root Voxel Count", 1: "Root Voxel Probability", 2: "Root Voxel Density"}
   
    norm = colors.Normalize(vmin=0, vmax=max_val) 
    plt.axis("off") 
    if PLOT_TITLE == True:
        plt.title(subtitle , fontsize=12)
        plt.suptitle(NAME, fontsize=16, fontweight='bold')
    else:
        plt.suptitle(subtitle, fontsize=16, fontweight='bold') 
        percent = (local_total / global_total) * 100
        plt.title(f"Depth containing {percent:.2f}% of annotated root voxels", fontsize=12)

    if PERCENT_SHOW == True and global_total is not None and local_total is not None and PLOT_TITLE == True:
        percent = (local_total / global_total) * 100
        ax.text(
            0.95, 0.05, f"Depth containing {percent:.2f}% of annotated root voxels",
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=12, color='black', backgroundcolor='white'
    )

    if save:
        os.makedirs(plot_path, exist_ok=True)
        full_path = os.path.join(plot_path, plot_name + plot_format)
        plt.savefig(full_path)
        print("Saved plot to:", full_path)

    return output_image 




Y, Z = create_mesh_grid(seg_vol, center_voxel, step_voxel=(1,1,1))

voxel_to_coord = list_voxel_to_coord(voxel_cylinder_edges, Y, Z)

radius_list, big_radius = make_large_radius(voxel_to_coord, center_voxel, NUM_CIRCLES)

label_circle = circle_levels_divider(radius_list, Y, Z)

angles_list = angular_slicing(NUM_ANGLES)

label_angle = angle_levels_divider(angles_list, Y, Z)

label_combined = combine_labels(label_circle, label_angle, NUM_ANGLES)

labels = [label_circle, label_angle, label_combined]

Areas = Area(NUM_ANGLES, NUM_CIRCLES, angles_list, radius_list)

summary_lines.append(f"  - Bin Areas: {Areas}")


depth = seg_vol.shape[0]
segment_size = depth // DEPTH_SEGMENTS
split_indices = [(i * segment_size, (i + 1) * segment_size) for i in range(DEPTH_SEGMENTS)]

if depth % DEPTH_SEGMENTS != 0:
    split_indices[-1] = (split_indices[-1][0], depth)

depth_splits = [seg_vol[start:end] for start, end in split_indices]


summed_vols = [np.sum(part, axis=0) for part in depth_splits]



global_total_count = np.sum(seg_vol)


global_max_vals = {}

for count_type in [0, 1, 2]:
    max_val = 0
    for summed_vol in summed_vols:
        values = return_values(
            count_type,
            summed_vol,
            labels[TYPE],
            ALL_LABELS,
            label_circle,
            Areas,
            global_total=global_total_count
        )
        max_val = max(max_val, np.max(values))
    global_max_vals[count_type] = max_val




for i, ((start, end), summed_vol) in enumerate(zip(split_indices, summed_vols)):
    for count_type in [0, 1, 2]:  # Count, Probability, Density
        count_label = count_dict[count_type]
        subtitle = f"Depth segment {i+1} ({start}-{end})"
        if count_type == 0:
            NAME= "Root Voxel Distribution " 
        elif count_type == 1:
            NAME= "Root Voxel Probability Distribution "
        elif count_type == 2:
            NAME= "Root Voxel Density Distribution "
        PLOT_PATH = SAMPLE + "_plots"
        PLOT_FORMAT = ".png"

        plot_name = f"{NAME}_{subtitle}_{count_label}_{NUM_CIRCLES}_{NUM_ANGLES}_{i}_{type_dict[TYPE]}_{COLORMAP}"
        
        local_total = np.sum(summed_vol)

        values = return_values(
            count_type,
            summed_vol,
            labels[TYPE],
            ALL_LABELS,
            label_circle,
            Areas,
            global_total=global_total_count
        )
        local_max_val = np.max(values)


        max_val = global_max_vals[count_type]

        max_index = np.argmax(values)
        percent_of_total = (local_total / global_total_count) * 100


        if count_type == 0:
            summary_lines.append(
                f"Depth {start}-{end}, {count_label}:\n"
                f"  -> Total root voxel count in depth: {local_total}\n"
                f"  -> Bin {max_index} has the most root voxels ({int(local_max_val )})\n"
            )
        elif count_type == 1:
            summary_lines.append(
                f"Depth {start}-{end}, {count_label}:\n"
                f"  -> This depth contains {percent_of_total:.2f}% of total root voxels\n"
                f"  -> Bin {max_index} has the highest probability ({local_max_val :.5f})\n"
            )
        elif count_type == 2:
            summary_lines.append(
                f"Depth {start}-{end}, {count_label}:\n"
                f"  -> Total root voxel count in depth: {local_total}\n"
                f"  -> Bin {max_index} is the most dense (density: {local_max_val })\n"
            )


        plot_cylinder_sectioning_single(
            labels[TYPE],
            label_circle,
            summed_vol,
            count_type,
            Areas,
            colormap=COLORMAP,
            save=True,
            plot_name=plot_name,
            plot_format=PLOT_FORMAT,
            plot_path=PLOT_PATH,
            global_total=global_total_count,
            local_total=local_total,
            max_val=max_val
        )



os.makedirs(PLOT_PATH, exist_ok=True)
report_path = os.path.join(PLOT_PATH, report_filename)

with open(report_path, "w") as f:
    for line in summary_lines:
        f.write(line + "\n")

print(f"Summary written to {report_path}")

print("Script finished")
