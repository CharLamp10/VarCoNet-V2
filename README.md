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
- HCP. Considering that there are two folders REST1 and REST2 at certain locations containing the downloaded files, one can run the data_preparations/parcellate_HCP.py script to extract the time-series from the ROIs of the selected atlas, giving as input the path to these folders, the path to the atlas and the output directory. Following, one should run the data_preparations/subsampling_HCP.py script to downsample data to a TR of 1.5 s, giving as input the path to the output directory from the previous script.
- ABIDE I. Considering there is a folder named fmriprep (downloaded from cyberduck) at a certain location and a .csv phenotypic file, one can run the data_preparations/parcellate_ABIDEI.py script to extract the time-series from the ROIs of the selected atlas, giving as input the path to the fmriprep folder, the path to the atlas, the path to the phenotypic file and the output directory.
- ABIDE II. Considering there is a folder named fmriprep (downloaded from cyberduck) at a certain location and a .csv phenotypic file, one can run the data_preparations/parcellate_ABIDEII.py script to extract the time-series from the ROIs of the selected atlas, giving as input the path to the fmriprep folder, the path to the atlas, the path to the phenotypic file and the output directory.

# Prepare data to be fed to the DL models
- HCP. Run the script data_preparations/prepare_HCP_data.py using as inputs the path of the output directory from the data_preparations/parcellate_HCP.py and an output directory
- ABIDE I. Run the script data_preparations/prepare_ABIDEI_data.py using as inputs the path of the output directory from the data_preparations/parcellate_ABIDEI.py and an output directory
- ABIDE II. Run the script data_preparations/prepare_ABIDEII_data.py using as inputs the path of the output directory from the data_preparations/parcellate_ABIDEII.py and an output directory

# Run Bayesian optimization to find suitable values for hyperparameters
One can run the VarCoNet_BO.py script. This script saves two .pkl files, one containing the best values for the examined hyperparameters and one containing information for all trials. This script requires the following inputs:
```python
config = {}
config['path_data'] = r'.../HCP'
config['atlas'] = 'AICHA' #AICHA, AAL
config['train_length_limits'] = [30,320]
config['test_lengths'] = [30,175,320]
config['test_num_winds'] = 10
config['shuffle'] = True
config['epochs'] = 200
config['warm_up_epochs'] = 10
config['eval_epochs'] = list(range(10,config['epochs']+1))
config['device'] = "cuda:0"
```

# Train and test VarCoNet for subject fingerprinting using HCP data
To apply VarCoNet on the HCP data one can run:
```python
python -m subject_fingerprinting \ 
  --path_data .../HCP \
  --path_save .../FOLDER \
  --atlas AICHA \
  --save_models \
  --save_results
```
There are additional input arguments that one can set. For more information check the script.

# Run ablations on subject fingerprinting
To run the ablation experiments one can run:
```python
python -m subject_fingerprinting_ablations \ 
  --path_data .../HCP \
  --path_save .../FOLDER \
  --save_results
```
There are additional input arguments that one can set. For more information check the script.

# Train and test VarCoNet for ASD classification using ABIDE I data
To apply VarCoNet on the ABIDE I data one can run:
```python
python -m ASD_classification_ABIDEI \ 
  --path_data .../ABIDEI \
  --path_save .../FOLDER \
  --atlas AICHA \
  --save_models \
  --save_results
```
There are additional input arguments that one can set. For more information check the script.

# Run ablations on ASD classification
To run the ablation experiments one can run:
```python
python -m ASD_classification_ablations \ 
  --path_data .../ABIDEI \
  --path_save .../FOLDER \
  --save_results
```
There are additional input arguments that one can set. For more information check the script.

# Train and test BolT on ASD classification using ABIDE I data
To apply BolT on the ABIDE I data one can run:
```python
python -m competing_methods.bolt \ 
  --path_data .../ABIDEI \
  --path_save .../FOLDER \
  --atlas AICHA \
  --save_models \
  --save_results
```
There are additional input arguments that one can set. For more information check the script. Other competing methods can also be run using a similar format.

# Exteral testing on ABIDE II
To test BolT and VarCoNet on ABIDE II run:
```python
python -m ASD_classification_ABIDEII \ 
  --path_data .../ABIDEII \
  --path_save .../FOLDER \ 
  --save_results
```

# Prediction stability
To calculate prediction stability please run:
```python
python -m predictions_stability \ 
  --path_data .../ABIDEI \
  --path_save .../FOLDER \ 
  --save_results
```
There are additional input arguments that one can set. For more information check the script. Other competing methods can also be run using a similar format.

**For all these scripts, it is important to use the same --path_save!!!**




