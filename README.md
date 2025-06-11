# Root Segmentation Pipeline - Bachelor Thesis

This repository contains the code accompanying the bachelor thesis Segmentation and Characterization of Plant Root System Architecture from Micro-CT Scans:

*"Accurate characterization of plant root system architecture in natural soil environments is essential for advancing plant science but remains technically challenging..."*

*(full abstract in thesis)*

The project presents a **proof-of-concept deep learning pipeline** for segmenting and analyzing plant root structures in **high-resolution micro-CT images** using a 3D UNet model. In addition to root segmentation, the pipeline includes methods for visual analysis of root architecture via **polar histogram heatmaps**.

---

## Repository Structure

```
├── Data/                     # Preprocessing and cropping
│   ├── Cropping.ipynb        # Jupyter notebook for interactive cropping
│   ├── down_sample.ipynb     # Jupyter notebook for downsampling volumes
│   └── preprocess_rawdata.py # Script for preprocessing raw data files
│
├── PostProcessing/           # Postprocessing and analysis
│   ├── histogram_analysis.py         # Histogram-based root analysis
│   ├── projection_heatmap.py         # Projection heatmap generation
│   └── other scripts/notebooks as needed
│
├── Training/                 # Training pipeline
│   ├── code/                 # Modular training code
│   │   ├── __init__.py
│   │   ├── main.py           # Main training entry point
│   │   ├── data_loader.py
│   │   ├── model.py
│   │   ├── loss.py
│   │   ├── trainer.py
│   │   └── (other modules as needed)
│   ├── configs/              # YAML configs controlling training
│   │   ├── example_config.yaml
│   │   └── (other configs)
│   └── logs/ (optional)      # Training logs, not included in Git repo
│
├── environment.yml           # Conda environment for reproducibility
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/root-segmentation-thesis.git
cd root-segmentation-thesis
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate root-segmentation-env  # or your exported env name
```

---

## Usage

### Preprocessing

Prepare raw micro-CT data by using the provided notebooks and preprocessing script in `Data/`.

### Training

Train a 3D UNet model using the modular training pipeline:

```bash
cd Training
python -m code.main --config configs/example_config.yaml
```

Training is controlled via YAML configuration files (`configs/`).

An example config looks like:

```yaml
train_dataset: down_1_res_week2_30_top_400_all
val_dataset: down_1_res_week2_30_top_400_all
train_transforms: "no_patch"
val_transforms: "no_patch"
learning_rate: 0.001
num_epochs: 1201
model:
  type: "UNet"
  in_channels: 1
  out_channels: 1
  channels: [16, 32, 64, 128, 256]
  strides: [2, 2, 2, 2]
loss:
  type: "dice"
```

### Postprocessing

After training, perform analysis using scripts in `PostProcessing/`:

- **Histogram analysis**: Quantify root architecture.
- **Projection heatmap**: Visualize and compare root growth patterns across samples and time points.

---

## Notes

- The pipeline is optimized for running on an HPC cluster with GPU support.
- Due to memory constraints of high-resolution micro-CT volumes, the training pipeline supports both **patch-based** and **full-volume** training and inference.
- PersistentDataset caching is optional and dependent on cluster filesystem performance.

---

## License

This project is licensed under the MIT License.
