# VarCoNet
Code developed and tested in Python 3.8.12 using PyTorch 2.1.2. Below are some necessary requirements.

```python
numpy == 1.26.4
scikit-learn == 1.5.2
optuna == 4.1.0
pytorch-lightning == 1.9.5
nibabel == 5.3.2
nilearn == 0.11.1
lightning-bolts == 0.7.0
pandas == 2.2.3
```

# Block diagram
![Architecture Diagram](plots/block_diagram.png)

# Download preprocessed fMRI data
- Preprocessed HCP data can be downloaded from the official webiste of the HCP: https://db.humanconnectome.org . One need to create an account to download the data. Once logged in, one can click the "Subjects with 3T MR session data" option at the "WU-Minn HCP Data - 1200 Subjects" panel and then select the following packages to download: Resting State fMRI 1 Preprocessed and Resting State fMRI 2 Preprocessed. These packages have size 6492.25 GB and 6112.31 GB, respectively. Phenotypic information can also be downloaded from the same site. Click the "WU-Minn HCP Data - 1200 Subjects" and then download "Behavioral Data" from Quick downloads
- Preprocessed ABIDE I data can be downloaded following the instructions in https://preprocessed-connectomes-project.org/abide/download.html . Specifically, one can download for free the cyberduck software (https://cyberduck.io/). Following, one can open cyberduck -> Open Connection -> FTP (choose Amazon S3) -> Access Key ID (anonymous) -> Path -> /fcp-indi/data/Projects/ABIDE/Outputs/fmriprep. One should see two folders, one named "fmriprep" and one named "freesurfer". One can proceed with downloading the "fmriprep" folder. Phenotypic information can also be downloaded from the cyberduck application (full path: /fcp-indi/data/Projects/ABIDE/Phenotypic_V1_0B_preprocessed1.csv)
- Preprocessed ABIDE II data can be downloaded following the exact same procedure as above but change the path to /fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep . Phenotypic information can be downloaded from https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html ("ABIDE II Composite Phenotypic File" under phenotypic data, under ABIDE II downloads).

# Download atlases
The two atlases (AICHA and AAL3) can be downloaded from (https://www.gin.cnrs.fr/en/tools/aicha/ and https://www.oxcns.org/aal3.html).

# Parcellate and filter fMRI data
- HCP. Considering that there are two folders REST1 and REST2 at certain locations containing the downloaded files, one can run the data_preparations/parcellate_HCP.py script to extract the time-series from the ROIs of the selected atlas, giving as input the path to these folders, and the path to the atlas. Following, one should run the data_preparations/subsampling_HCP.py script to downsample data to a TR of 1.5 s.
- ABIDE I. Considering there is a folder named fmriprep (downloaded from cyberduck) at a certain location and a .csv phenotypic file, one can run the data_preparations/parcellate_ABIDEI.py script to extract the time-series from the ROIs of the selected atlas, giving as input the path to the fmriprep folder, the path to the atlas and the path to the phenotypic file.
- ABIDE II. Considering there is a folder named fmriprep (downloaded from cyberduck) at a certain location and a .csv phenotypic file, one can run the data_preparations/parcellate_ABIDEII.py script to extract the time-series from the ROIs of the selected atlas, giving as input the path to the fmriprep folder, the path to the atlas and the path to the phenotypic file.


