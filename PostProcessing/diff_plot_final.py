# Libraries
import qim3d
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors
from skimage.segmentation import find_boundaries

print("Difference Histogram")

# Sample specifications
SAMPLE = "Dif_Sample_30_Week_2_and_Week_3_final"
SAMPLE_PATH_1 = "QIM/projects/2024_DANFIX_147_RAPID-P/analysis/sample30/s30_joint/week2-joint-root-class.nii.gz"
SAMPLE_PATH_2 = "QIM/projects/2024_DANFIX_147_RAPID-P/analysis/sample30/s30_joint/week3-joint-root-class.nii.gz"

# Image-specific variables
center_voxel = (388, 223, 223)
voxel_cylinder_edges = [(15, 223), (430, 223), (223, 25), (223, 440)]

# Plot variables
NUM_CIRCLES = 7
NUM_ANGLES = 7
DEPTH_SEGMENTS = 3
TYPE = 2
ALL_LABELS = np.arange(NUM_CIRCLES * NUM_ANGLES)

USE_GLOBAL_COLORBAR_RANGE = True
SHOW_COLORBAR = True
count_dict = {0: "count", 1: "probability", 2: "density"}
COLORMAP = "seismic"
BORDER_COLOR = [0, 0, 0]
BACKGROUND_COLOR = [255, 255, 255]
PLOT_TITLE = False
PERCENT_SHOW = True

def create_mesh_grid(volume, center_voxel):
    y = np.arange(volume.shape[1])
    z = np.arange(volume.shape[2])
    Y, Z = np.meshgrid(y, z, indexing='ij')
    return Y - center_voxel[1], Z - center_voxel[2]

def make_large_radius(edges, center, num):
    radii = [np.sqrt((x - center[1])**2 + (y - center[2])**2) for x, y in edges]
    avg_radius = np.mean(radii)
    return [(i + 1) * avg_radius / num for i in range(num)], avg_radius

def circle_labels(Y, Z, radii):
    R = np.sqrt(Y**2 + Z**2)
    labels = np.searchsorted(radii, R)
    labels[R > radii[-1]] = -1
    return labels

def angle_labels(Y, Z, num):
    angles = (np.degrees(np.arctan2(Z, Y)) + 360) % 360
    step = 360 / num
    return np.floor(angles / step).astype(int)

def combine_labels(circ, angle, num_angles):
    return circ * num_angles + angle

def compute_areas(num_circ, num_angle, radii):
    areas = []
    for i in range(num_circ):
        r_outer = radii[i]
        r_inner = 0 if i == 0 else radii[i - 1]
        area_ring = (np.pi * (r_outer**2 - r_inner**2)) / num_angle
        areas.extend([area_ring] * num_angle)
    return np.array(areas)

def compute_bin_counts(vol, labels, mask):
    return np.array([np.sum(vol[(labels == i) & mask]) for i in ALL_LABELS])

def compute_probability(counts, total):
    return counts / total if total > 0 else np.zeros_like(counts)

def compute_density(probs, areas):
    return probs / (areas + 1e-8)

def plot_diff(values, labels, circle_mask, min_val, max_val, subtitle, name, total, local, count_type, save_path, show_colorbar=True):
    cmap = plt.get_cmap(COLORMAP)
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=min_val, vmax=max_val)
    rgb = (np.array([cmap(norm(v))[:3] for v in values]) * 255).astype(np.uint8)

    out_img = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i in ALL_LABELS:
        out_img[labels == i] = rgb[i]

    boundaries = find_boundaries(labels, mode='outer')
    out_img[boundaries] = BORDER_COLOR
    out_img[circle_mask < 0] = BACKGROUND_COLOR

    fig, ax = plt.subplots()
    ax.imshow(out_img)
    ax.axis("off")

    if show_colorbar:
        label_dict = ["Count Δ", "Probability Δ", "Density Δ"]
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label(label_dict[count_type])

    if PLOT_TITLE == True:
        plt.title(subtitle , fontsize=12)
        plt.suptitle(NAME, fontsize=16, fontweight='bold')
    else:
        plt.suptitle(subtitle, fontsize=16, fontweight='bold') 
        percent = (local / total) * 100
        plt.title(f"Depth containing {percent:.2f}% of annotated root voxels", fontsize=12)

    os.makedirs(save_path, exist_ok=True)

    suffix = "" if show_colorbar else "_no_bar"
    plt.savefig(os.path.join(save_path, f"{name}{suffix}_{count_type}.png"))
    plt.close()

# Load volumes
id = str(Path.home()) + "/"
seg1, _ = qim3d.io.load(id + SAMPLE_PATH_1, return_metadata=True)
seg1[seg1 == 2] = 0
seg1 = seg1 > 0

seg2, _ = qim3d.io.load(id + SAMPLE_PATH_2, return_metadata=True)
seg2[seg2 == 2] = 0
seg2 = seg2 > 0

# Setup mesh and labels
Y, Z = create_mesh_grid(seg1, center_voxel)
radii, _ = make_large_radius(voxel_cylinder_edges, center_voxel, NUM_CIRCLES)
circle = circle_labels(Y, Z, radii)
angle = angle_labels(Y, Z, NUM_ANGLES)
combined = combine_labels(circle, angle, NUM_ANGLES)
Areas = compute_areas(NUM_CIRCLES, NUM_ANGLES, radii)

depth = seg1.shape[0]
split = [(i * (depth // DEPTH_SEGMENTS), (i + 1) * (depth // DEPTH_SEGMENTS)) for i in range(DEPTH_SEGMENTS)]
split[-1] = (split[-1][0], depth)

total1 = np.sum(seg1)
total2 = np.sum(seg2)
total_diff = total2 - total1

summary_lines = []
plot_path = SAMPLE + "_plots"

summary_lines.append("Plot settings:")
summary_lines.append(f"  - Number of circles: {NUM_CIRCLES}")
summary_lines.append(f"  - Number of angular bins: {NUM_ANGLES}")
summary_lines.append(f"  - Number of depth segments: {DEPTH_SEGMENTS}\n")
summary_lines.append("Image-specific points:")
summary_lines.append(f"  - Center voxel: {center_voxel}")
summary_lines.append(f"  - Voxel cylinder edges: {voxel_cylinder_edges}\n")


global_max_vals = {0: 0, 1: 0, 2: 0}

for i, (start, end) in enumerate(split):
    d1 = np.sum(seg1[start:end], axis=0)
    d2 = np.sum(seg2[start:end], axis=0)
    mask = circle != -1
    labels = [circle, angle, combined][TYPE]

    c1 = compute_bin_counts(d1, labels, mask)
    c2 = compute_bin_counts(d2, labels, mask)

    p1 = compute_probability(c1, total1)
    p2 = compute_probability(c2, total2)

    dens1 = compute_density(p1, Areas)
    dens2 = compute_density(p2, Areas)

    diffs = [c2 - c1, p2 - p1, dens2 - dens1]

    for count_type, diff in enumerate(diffs):
        max_abs = np.max(np.abs(diff))
        if max_abs > global_max_vals[count_type]:
            global_max_vals[count_type] = max_abs

for i, (start, end) in enumerate(split):
    d1 = np.sum(seg1[start:end], axis=0)
    d2 = np.sum(seg2[start:end], axis=0)

    mask = circle != -1
    labels = [circle, angle, combined][TYPE]

    c1 = compute_bin_counts(d1, labels, mask)
    c2 = compute_bin_counts(d2, labels, mask)
    p1 = compute_probability(c1, total1)
    p2 = compute_probability(c2, total2)
    dens1 = compute_density(p1, Areas)
    dens2 = compute_density(p2, Areas)

    diffs = [c2 - c1, p2 - p1, dens2 - dens1]
    bases = [c2, p2, dens2]
    local_total = np.sum(d2) - np.sum(d1)

    for count_type, (vals, diff) in enumerate(zip(bases, diffs)):
        if USE_GLOBAL_COLORBAR_RANGE:
            vmin = -global_max_vals[count_type]
            vmax = global_max_vals[count_type]
            suffix = "_global"
        else:
            vmin = -np.max(np.abs(diff))
            vmax = np.max(np.abs(diff))
            suffix = ""
        subtitle = f"Depth segment {i + 1} ({start}-{end})"
        NAME = {
            0: "Root Voxel Distribution",
            1: "Root Voxel Probability Distribution",
            2: "Root Voxel Density Distribution"
        }[count_type]
        name = f"{NAME}{suffix}_{start}-{end}_{count_type}"
        print(f"Generating plot for depth {i} ({start}-{end}), count_type {count_type}, file: {NAME + suffix}_{count_type}.png")
        plot_diff(diff, labels, circle, vmin, vmax, subtitle, name, total_diff, local_total, count_type, plot_path, show_colorbar=SHOW_COLORBAR)


        if count_type == 0:
            summary_lines.append(f"Depth {start}-{end}, {count_dict[count_type]}:")
            summary_lines.append(f"  -> Total root voxel count in depth: {int(local_total)}")
            summary_lines.append(f"  -> Bin {np.argmax(vals)} has the most root voxels ({int(np.max(vals))})\n")
        elif count_type == 1:
            summary_lines.append(f"Depth {start}-{end}, {count_dict[count_type]}:")
            summary_lines.append(f"  -> This depth contains {(local_total / total_diff * 100):.2f}% of total root voxels")
            summary_lines.append(f"  -> Bin {np.argmax(vals)} has the highest probability ({np.max(vals):.5f})\n")
        elif count_type == 2:
            summary_lines.append(f"Depth {start}-{end}, {count_dict[count_type]}:")
            summary_lines.append(f"  -> Total root voxel count in depth: {int(local_total)}")
            summary_lines.append(f"  -> Bin {np.argmax(vals)} is the most dense (density: {np.max(vals)})\n")

with open(os.path.join(plot_path, f"Dif_report_{SAMPLE}.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("Summary written")
print("Done")
