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

# fMRI preprocessing and parcellation
- Preprocessed HCP data can be downloaded from the official webiste of the HCP: https://db.humanconnectome.org . One need to create an account to download the data. Once logged in, one can click the "Subjects with 3T MR session data" option at the "WU-Minn HCP Data - 1200 Subjects" panel and then select the following packages to download: Resting State fMRI 1 Preprocessed and Resting State fMRI 2 Preprocessed. These packages have size 6492.25 GB and 6112.31 GB, respectively
- Preprocessed ABIDE I data can be downloaded following the instructions in https://preprocessed-connectomes-project.org/abide/download.html . Specifically, one can download for free the cyberduck software (https://cyberduck.io/). Following, one can open cyberduck -> Open Connection -> FTP (choose Amazon S3) -> Access Key ID (anonymous) -> Path -> /fcp-indi/data/Projects/ABIDE/Outputs/fmriprep. One should see two folders, one named "fmriprep" and one named "freesurfer". One can proceed with downloading the "fmriprep" folder.
- Follow the exact same procedure as above but change the path to /fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep
.
