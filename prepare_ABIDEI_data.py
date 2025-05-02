import numpy as np
import os
from scipy.signal import resample
import pandas as pd


def resample_signal(signal,file,site):
    if 'Caltech' in file or 'CMU_a' in file or 'SDSU' in file or 'Stanford' in file or 'UM' in file or 'Yale' in file or 'Trinity' in file or 'USM' in file or 'NYU' in file:
        TR = 2.0
    elif 'KKI' in file or 'OHSU' in file:
        TR = 2.5
    elif 'Leuven' in file:
        TR = 1.6669999999999998
    elif 'MaxMun' in file or 'UCLA' in file:
        TR = 3
    elif 'Olin' in file or 'Pitt' in file or 'CMU_b' in file:
        TR = 1.5
    elif 'SBL' in file:
        TR = 2.2
    elif 'no_filename' in file:
        if 'CALTECH' in site or 'SDSU' in site or 'STANFORD' in site or 'UM' in site or 'YALE' in site or 'TRINITY' in site or 'USM' in site or 'NYU' in site:
            TR = 2.0
        elif 'KKI' in site or 'OHSU' in site:
            TR = 2.5
        elif 'LEUVEN' in site:
            TR = 1.6669999999999998
        elif 'MAX_MUN' in site or 'UCLA' in site:
            TR = 3
        elif 'OLIN' in site or 'PITT' in site:
            TR = 1.5
        elif 'SBL' in site:
            TR = 2.2
    else:
        raise Exception('Did not find correspondance')
    samples = signal.shape[0]
    duration = samples*TR/60
    if TR > 1.5:
        ratio = TR/1.5
        new_len = int(np.round(signal.shape[0]*ratio))
        signal = resample(signal,new_len)
    return signal,duration

max_size = 320
path_data_AICHA = r'E:\ABIDEI_AICHA_AAL_nilearn\fmriprep\ROISignals_AICHA'
path_data_AAL = r'E:\ABIDEI_AICHA_AAL_nilearn\fmriprep\ROISignals_AAL'
save_path = r'C:\Users\100063082\Desktop\SSL_FC_matrix_GNN_data\ABIDE\fmriprep'
path_phenotypic = r'C:\Users\100063082\Desktop\prepare_ADHD200_ABIDE_phenotypics\Phenotypic_V1_0b_preprocessed1.csv'


data_dir = os.listdir(path_data_AICHA)


phenotypic = pd.read_csv(path_phenotypic)
sub_id = phenotypic['SUB_ID'].values
group = phenotypic['DX_GROUP'].values
file_ids = phenotypic['FILE_ID'].values
sites = phenotypic['SITE_ID'].values
age = phenotypic['AGE_AT_SCAN'].values
gender = phenotypic['SEX'].values
group[group == 2] = 0

names = []
Times_AICHA = []
Times_AAL = []
classes = []
ages = []
genders = []
durations = []
for data in data_dir:
    pos00 = data.find('00')
    pos_task = data.find('_task')
    ID = data[pos00+2:pos_task]
    pos_id = np.where(sub_id == int(ID))[0]
    if len(pos_id) != 1:
        raise TypeError("ID false identification")
    feature_dir_aicha = os.path.join(path_data_AICHA, data,'ROISignals.npy')
    feature_dir_aal = os.path.join(path_data_AAL, data,'ROISignals.npy')
    feature_aicha = np.load(feature_dir_aicha)#np.array(sio.loadmat(feature_dir_aicha)['ROISignals'])
    feature_aal = np.load(feature_dir_aal)#np.array(sio.loadmat(feature_dir_aal)['ROISignals'])
    if not np.any(np.all(feature_aicha == 0, axis=0)) and not np.any(np.all(feature_aal == 0, axis=0)):
        file_id = file_ids[pos_id[0]]
        site = sites[pos_id[0]]
        classes.append(group[pos_id[0]])
        ages.append(age[pos_id[0]])
        genders.append(gender[pos_id[0]])
        feature_aicha,duration = resample_signal(feature_aicha, file_id, site)          
        feature_aal,_ = resample_signal(feature_aal, file_id, site)   
        durations.append(duration)
        if feature_aicha.shape[0] > max_size:
            feature_aicha = feature_aicha[:max_size,:]
            feature_aal = feature_aal[:max_size,:]
        elif feature_aicha.shape[0] < max_size:
            feature_aicha = np.concatenate((feature_aicha,np.zeros((max_size-feature_aicha.shape[0],feature_aicha.shape[1]))),axis=0)
            feature_aal = np.concatenate((feature_aal,np.zeros((max_size-feature_aal.shape[0],feature_aal.shape[1]))),axis=0)
        Times_AICHA.append(feature_aicha)
        Times_AAL.append(feature_aal)
        names.append(data)
    else:
        print(data)

#subIDs = []
#for name in names:
#    subIDs.append(name[:11])
#counts = []
#for subID in subIDs:
#    if len(np.where(np.array(subIDs) == subID)[0]) > 1:
#        counts.append(0)
#    else:
#        counts.append(1)
#ASD_inds = np.where(((np.array(classes) == 1)*1)*counts)[0]
#NC_inds = np.where(((np.array(classes) == 0)*1)*counts)[0]  
          
#test_inds = list(ASD_inds[sample(range(len(ASD_inds)), 50)]) + list(NC_inds[sample(range(len(NC_inds)), 50)])
#test = [data_dir[i] for i in test_inds]
    
#train_inds = [item for item in range(len(Times_AAL)) if item not in test_inds]
#train = [item for item in data_dir if item not in test]
#for i,name in enumerate(train):
#    train[i] = name[:11]
#for i,name in enumerate(test):
#    test[i] = name[:11]
#Time_train_AICHA = [Times_AICHA[i] for i in train_inds]
#Time_test_AICHA = [Times_AICHA[i] for i in test_inds]
#Time_train_AAL = [Times_AAL[i] for i in train_inds]
#Time_test_AAL = [Times_AAL[i] for i in test_inds]
#class_train = [classes[i] for i in train_inds]
#class_test = [classes[i] for i in test_inds]



#class_train = np.array(class_train)   
#class_test = np.array(class_test) 
classes = np.array(classes) 
#np.save(os.path.join(save_path,'ABIDE_class_train.npy'),class_train)
#np.save(os.path.join(save_path,'ABIDE_class_test.npy'),class_test) 
np.save(os.path.join(save_path,'ABIDE_nilearn_classes.npy'),classes) 
#np.savez(os.path.join(save_path,'ABIDE_train_AICHA'),*Time_train_AICHA)
#np.savez(os.path.join(save_path,'ABIDE_train_AAL'),*Time_train_AAL) 
#np.savez(os.path.join(save_path,'ABIDE_test_AICHA'),*Time_test_AICHA)
#np.savez(os.path.join(save_path,'ABIDE_test_AAL'),*Time_test_AAL) 
np.savez(os.path.join(save_path,'ABIDE_nilearn_AICHA'),*Times_AICHA)
np.savez(os.path.join(save_path,'ABIDE_nilearn_AAL'),*Times_AAL) 
#with open(os.path.join(save_path,'ABIDE_names_train.txt'), 'w') as f:
#    for item in train:
#        f.write("%s\n" % item)
#with open(os.path.join(save_path,'ABIDE_names_test.txt'), 'w') as f:
#    for item in test:
#        f.write("%s\n" % item)
for i,name in enumerate(names):
    names[i] = name[:11]
with open(os.path.join(save_path,'ABIDE_nilearn_names.txt'), 'w') as f:
    for item in names:
        f.write("%s\n" % item)
