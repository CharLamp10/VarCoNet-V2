import os
from nilearn.input_data import  NiftiLabelsMasker
import pandas as pd
import numpy as np

#FIX IF CONFOUNDS IS MISSING
atlas = 'AAL'

if atlas == "AAL":
    path_atlas = r'C:\Users\100063082\Desktop\spm12\toolbox\AAL3\AAL3v1.nii'
else:
    path_atlas = r'C:\Users\100063082\Desktop\spm12\toolbox\AICHA\AICHA.nii'


path = r'D:\ABIDEII\fmriprep'
path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\ABIDEII_Composite_Phenotypic.csv'
phenotypic = pd.read_csv(path_phenotypic,encoding="latin-1")
ids = phenotypic["SUB_ID"].values
sites = phenotypic["SITE_ID"].values

path_save = r'E:\ABIDEII_AICHA_AAL_nilearn\fmriprep\ROIsignals_' + atlas
if not os.path.exists(path_save):
    os.makedirs(path_save,exist_ok=True)

dire = os.listdir(path);
for i in range(len(dire)):
    if 'sub' in dire[i] and os.path.isdir(os.path.join(path,dire[i])):
        path_subject = os.path.join(path,dire[i])
        dir_sucject = os.listdir(path_subject)
        for ses in dir_sucject:
            if 'ses' in ses:
                dir_session = os.listdir(os.path.join(path_subject, ses))
                path_func = os.path.join(path_subject, ses, 'func')
                if os.path.exists(path_func):
                    dir_func = os.listdir(path_func)
                    flag = 0
                    for j in range(len(dir_func)):
                        file = dir_func[j]
                        if 'smoothAROMAnonaggr' in file:
                            flag = flag + 1
                        if 'confounds' in file:
                            flag = flag + 1
                    if flag == 2:
                        for j in range(len(dir_func)):
                            file = dir_func[j]
                            if 'smoothAROMAnonaggr' in file:
                                pos_space = file.find("_space")   
                                pos_underscore = file.find("_")
                                path_data = os.path.join(path_func,file)
                                path_confounds = os.path.join(path_func,file[:pos_space] + '_desc-confounds_regressors.tsv')
                                confounds = pd.read_table(path_confounds).iloc[:,:2]
                                if pos_underscore == -1:
                                    raise ValueError("Invalid filename format")
                                
                                try:
                                    id_val = int(file[4:pos_underscore])
                                except ValueError:
                                    raise ValueError("Invalid ID extraction")
                                
                                pos_id = np.where(ids == id_val)[0]
                                
                                if len(pos_id) == 0:
                                    continue
                                
                                pos_id = pos_id[0]
                                site = sites[pos_id]
                                
                                tr_values = {
                                    "BNI": 3.0, "EMC": 2.0, "ETH": 2.0, "GU": 2.0, "IP": 2.7, 
                                    "IU": 0.813, "KKI": 2.5, "KUL": 2.5, "NYU": 2.0, "OHSU": 2.5, "OILH": 0.475, 
                                    "SDSU": 2.0, "SU": 2.0, "TCD": 2.0, "UCD": 2.0, "UCLA": 3.0, "USM": 2.0, "MIA": 2.0
                                }
                            
                                for key, tr in tr_values.items():
                                    if key in site:
                                        TR = tr
                                
                                
                                path_save_new = os.path.join(path_save,file[:pos_space])
                                if not os.path.exists(path_save_new):
                                    os.mkdir(path_save_new)
                            
                                if os.path.isfile(path_data):
                                    try:
                                        masker = NiftiLabelsMasker(
                                                labels_img=path_atlas,
                                                memory="nilearn_cache",
                                                low_pass=0.1,
                                                high_pass=0.01,
                                                t_r=TR,
                                                verbose=0,
                                            )
                                        time_series = masker.fit_transform(path_data, confounds=confounds)
                                        np.save(os.path.join(path_save_new,'ROISignals.npy'), time_series)
                                        print(file)
                                    except:
                                        continue
