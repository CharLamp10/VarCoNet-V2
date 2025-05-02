import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score,roc_curve
import pandas as pd
import seaborn as sns

if not os.path.exists(os.path.join("ablations","plots")):
    os.makedirs(os.path.join("ablations","plots"), exist_ok=True)

path_results = r'/home/student1/Desktop/Charalampos_Lamprou/VarCoNetV2'
    
with open(os.path.join(path_results,'results_ABIDE','AICHA','ABIDE_VarCoNetV2_results.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aicha = pickle.load(f)
with open(os.path.join(path_results,'results_ABIDE','AAL','ABIDE_VarCoNetV2_results.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aal = pickle.load(f)
with open(os.path.join(path_results,'ablations','results_ABIDE','AICHA','ABIDE_VarCoNetV2_noCNN_results.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aicha_noCNN = pickle.load(f)
with open(os.path.join(path_results,'ablations','results_ABIDE','AAL','ABIDE_VarCoNetV2_noCNN_results.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aal_noCNN = pickle.load(f)
with open(os.path.join(path_results,'results_ABIDE','AICHA','ABIDE_SL_results.pkl'), 'rb') as f:
    test_result_SL_aicha = pickle.load(f) 
with open(os.path.join(path_results,'results_ABIDE','AAL','ABIDE_SL_results.pkl'), 'rb') as f:
    test_result_SL_aal = pickle.load(f) 
    

val_losses = np.array(test_result_SL_aicha['val_losses'])
test_losses = np.array(test_result_SL_aicha['test_losses'])
test_aucs = np.array(test_result_SL_aicha['test_aucs'])
test_probs = test_result_SL_aicha['test_probs']
y_tests = test_result_SL_aicha['y_test']
min_indices = np.argmin(val_losses, axis=1)

SL_AICHA_test_f1 = np.zeros((len(val_losses),))
for i in range(len(val_losses)):
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    SL_AICHA_test_f1[i] = f1_score(y_test, test_prob)

SL_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
SL_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 


val_losses = np.array(test_result_SL_aal['val_losses'])
test_losses = np.array(test_result_SL_aal['test_losses'])
test_aucs = np.array(test_result_SL_aal['test_aucs'])
test_probs = test_result_SL_aal['test_probs']
y_tests = test_result_SL_aal['y_test']
min_indices = np.argmin(val_losses, axis=1)

SL_AAL_test_f1 = np.zeros((len(val_losses),))
for i in range(len(val_losses)):
    test_prob = test_probs[i][min_indices[i]]
    y_test = y_tests[i][min_indices[i]]
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    best_threshold = thresholds[best_index]
    test_prob[test_prob<best_threshold] = 0
    test_prob[test_prob>=best_threshold] = 1
    SL_AAL_test_f1[i] = f1_score(y_test, test_prob)

SL_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
SL_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((100,50))   
test_aucs = np.zeros((100,50))  
test_f1 = np.zeros((100,50))  
for i,test in enumerate(test_result_VarCoNetV2_aicha['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
        test_aucs[i,j] = test1['best_test_auc']
        test_losses[i,j] = test1['best_test_loss']
        test_prob = test1['test_probs']
        y_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        test_f1[i,j] = f1_score(y_test, test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
        
aicha_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
aicha_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
aicha_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((100,50))   
test_aucs = np.zeros((100,50))  
test_f1 = np.zeros((100,50))  
for i,test in enumerate(test_result_VarCoNetV2_aal['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
        test_aucs[i,j] = test1['best_test_auc']
        test_losses[i,j] = test1['best_test_loss']
        test_prob = test1['test_probs']
        y_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        test_f1[i,j] = f1_score(y_test, test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
        
aal_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
aal_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
aal_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((100,50))   
test_aucs = np.zeros((100,50))  
test_f1 = np.zeros((100,50))  
for i,test in enumerate(test_result_VarCoNetV2_aicha_noCNN['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
        test_aucs[i,j] = test1['best_test_auc']
        test_losses[i,j] = test1['best_test_loss']
        test_prob = test1['test_probs']
        y_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        test_f1[i,j] = f1_score(y_test, test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
        
noCNN_AICHA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
noCNN_AICHA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
noCNN_AICHA_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


test_losses = np.zeros((100,50))   
test_aucs = np.zeros((100,50))  
test_f1 = np.zeros((100,50))  
for i,test in enumerate(test_result_VarCoNetV2_aal_noCNN['epoch_results']):
    for j,test1 in enumerate(test):
        val_losses[i,j] = test1['best_val_loss']
        test_aucs[i,j] = test1['best_test_auc']
        test_losses[i,j] = test1['best_test_loss']
        test_prob = test1['test_probs']
        y_test = test1['y_test']
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
        youden_j = tpr - fpr
        best_index = np.argmax(youden_j)
        best_threshold = thresholds[best_index]
        test_prob[test_prob<best_threshold] = 0
        test_prob[test_prob>=best_threshold] = 1
        test_f1[i,j] = f1_score(y_test, test_prob)
        
min_indices = np.argmin(val_losses, axis=1)
        
noCNN_AAL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
noCNN_AAL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
noCNN_AAL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 


plt.figure()
y = np.concatenate((aal_test_losses, aicha_test_losses,
                    aal_test_aucs, aicha_test_aucs,
                    aal_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((200,), 'loss'),np.full((200,), 'AUC'),np.full((200,), 'F1')),axis=0)
hue = np.concatenate((np.full((100,), 'AAL'),np.full((100,), 'AICHA'),
                      np.full((100,), 'AAL'),np.full((100,), 'AICHA'),
                      np.full((100,), 'AAL'),np.full((100,), 'AICHA')),axis=0)

d = {'Score': y, 'Atlas': hue, 'Metric': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Metric", y="Score", hue="Atlas")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","ABIDE_atlas.png"), dpi=600, bbox_inches='tight')
plt.close()


plt.figure()
y = np.concatenate((noCNN_AICHA_test_losses, aicha_test_losses,
                    noCNN_AICHA_test_aucs, aicha_test_aucs,
                    noCNN_AICHA_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((200,), 'loss'),np.full((200,), 'AUC'),np.full((200,), 'F1')),axis=0)
hue = np.concatenate((np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes')),axis=0)

d = {'Score': y, 'With CNN': hue, 'Metric': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Metric", y="Score", hue="With CNN")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","ABIDE_noCNN_AICHA.png"), dpi=600, bbox_inches='tight')
plt.close()


plt.figure()
y = np.concatenate((noCNN_AAL_test_losses, aicha_test_losses,
                    noCNN_AAL_test_aucs, aicha_test_aucs,
                    noCNN_AAL_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((200,), 'loss'),np.full((200,), 'AUC'),np.full((200,), 'F1')),axis=0)
hue = np.concatenate((np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes')),axis=0)

d = {'Score': y, 'With CNN': hue, 'Metric': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Metric", y="Score", hue="With CNN")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","ABIDE_noCNN_AAL.png"), dpi=600, bbox_inches='tight')
plt.close()


plt.figure()
y = np.concatenate((SL_AICHA_test_losses, aicha_test_losses,
                    SL_AICHA_test_aucs, aicha_test_aucs,
                    SL_AICHA_test_f1, aicha_test_f1), axis=0)
x = np.concatenate((np.full((200,), 'loss'),np.full((200,), 'AUC'),np.full((200,), 'F1')),axis=0)
hue = np.concatenate((np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes')),axis=0)

d = {'Score': y, 'SSL': hue, 'Metric': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Metric", y="Score", hue="SSL")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","ABIDE_SSL_AICHA.png"), dpi=600, bbox_inches='tight')
plt.close()


plt.figure()
y = np.concatenate((SL_AAL_test_losses, aal_test_losses,
                    SL_AAL_test_aucs, aal_test_aucs,
                    SL_AAL_test_f1, aal_test_f1), axis=0)
x = np.concatenate((np.full((200,), 'loss'),np.full((200,), 'AUC'),np.full((200,), 'F1')),axis=0)
hue = np.concatenate((np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes'),
                      np.full((100,), 'No'),np.full((100,), 'Yes')),axis=0)

d = {'Score': y, 'SSL': hue, 'Metric': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Metric", y="Score", hue="SSL")
plt.ylabel('Score', fontsize=17)
plt.xlabel('Metric', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","ABIDE_SSL_AAL.png"), dpi=600, bbox_inches='tight')
plt.close()



with open(os.path.join(path_results,'ablations','results','test_results_AAL_VarCoNetV2_noCNN.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aal_noCNN = pickle.load(f)
with open(os.path.join(path_results,'ablations','results','test_results_AICHA_VarCoNetV2_noCNN.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aicha_noCNN = pickle.load(f)
with open(os.path.join(path_results,'results','test_results_AAL_VarCoNetV2.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aal = pickle.load(f)
with open(os.path.join(path_results,'results','test_results_AICHA_VarCoNetV2.pkl'), 'rb') as f:
    test_result_VarCoNetV2_aicha = pickle.load(f)   
    
AAL_0_0 = test_result_VarCoNetV2_aal['test_result'][0][0]
AAL_0_1 = test_result_VarCoNetV2_aal['test_result'][0][1]
AAL_0_2 = test_result_VarCoNetV2_aal['test_result'][0][2]
AAL_1_1 = test_result_VarCoNetV2_aal['test_result'][0][3]
AAL_1_2 = test_result_VarCoNetV2_aal['test_result'][0][4]
AAL_2_2 = test_result_VarCoNetV2_aal['test_result'][0][5]

AAL_noCNN_0_0 = test_result_VarCoNetV2_aal_noCNN['test_result'][0][0]
AAL_noCNN_0_1 = test_result_VarCoNetV2_aal_noCNN['test_result'][0][1]
AAL_noCNN_0_2 = test_result_VarCoNetV2_aal_noCNN['test_result'][0][2]
AAL_noCNN_1_1 = test_result_VarCoNetV2_aal_noCNN['test_result'][0][3]
AAL_noCNN_1_2 = test_result_VarCoNetV2_aal_noCNN['test_result'][0][4]
AAL_noCNN_2_2 = test_result_VarCoNetV2_aal_noCNN['test_result'][0][5]


plt.figure()
y = np.concatenate((AAL_noCNN_0_0, AAL_noCNN_0_1, AAL_noCNN_0_2,
                    AAL_noCNN_1_1, AAL_noCNN_1_2, AAL_noCNN_2_2,
                    AAL_0_0, AAL_0_1, AAL_0_2, AAL_1_1, AAL_1_2, AAL_2_2), axis=0)
x = np.concatenate((np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),),axis=0)
hue = np.concatenate((np.full((60,), 'No'), np.full((60,), 'Yes')),axis=0)

d = {'Fingerprinting Accuracy': y, 'With CNN': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With CNN")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","HCP_noCNN_AAL.png"), dpi=600, bbox_inches='tight')
plt.close()


AICHA_0_0 = test_result_VarCoNetV2_aicha['test_result'][0][0]
AICHA_0_1 = test_result_VarCoNetV2_aicha['test_result'][0][1]
AICHA_0_2 = test_result_VarCoNetV2_aicha['test_result'][0][2]
AICHA_1_1 = test_result_VarCoNetV2_aicha['test_result'][0][3]
AICHA_1_2 = test_result_VarCoNetV2_aicha['test_result'][0][4]
AICHA_2_2 = test_result_VarCoNetV2_aicha['test_result'][0][5]

AICHA_noCNN_0_0 = test_result_VarCoNetV2_aicha_noCNN['test_result'][0][0]
AICHA_noCNN_0_1 = test_result_VarCoNetV2_aicha_noCNN['test_result'][0][1]
AICHA_noCNN_0_2 = test_result_VarCoNetV2_aicha_noCNN['test_result'][0][2]
AICHA_noCNN_1_1 = test_result_VarCoNetV2_aicha_noCNN['test_result'][0][3]
AICHA_noCNN_1_2 = test_result_VarCoNetV2_aicha_noCNN['test_result'][0][4]
AICHA_noCNN_2_2 = test_result_VarCoNetV2_aicha_noCNN['test_result'][0][5]


plt.figure()
y = np.concatenate((AICHA_noCNN_0_0, AICHA_noCNN_0_1, AICHA_noCNN_0_2,
                    AICHA_noCNN_1_1, AICHA_noCNN_1_2, AICHA_noCNN_2_2,
                    AICHA_0_0, AICHA_0_1, AICHA_0_2, AICHA_1_1, AICHA_1_2, AICHA_2_2), axis=0)
x = np.concatenate((np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),),axis=0)
hue = np.concatenate((np.full((60,), 'No'), np.full((60,), 'Yes')),axis=0)

d = {'Fingerprinting Accuracy': y, 'With CNN': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="With CNN")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","HCP_noCNN_AICHA.png"), dpi=600, bbox_inches='tight')
plt.close()


AAL_0_0 = test_result_VarCoNetV2_aal['test_result'][0][0]
AAL_0_1 = test_result_VarCoNetV2_aal['test_result'][0][1]
AAL_0_2 = test_result_VarCoNetV2_aal['test_result'][0][2]
AAL_1_1 = test_result_VarCoNetV2_aal['test_result'][0][3]
AAL_1_2 = test_result_VarCoNetV2_aal['test_result'][0][4]
AAL_2_2 = test_result_VarCoNetV2_aal['test_result'][0][5]

AICHA_0_0 = test_result_VarCoNetV2_aicha['test_result'][0][0]
AICHA_0_1 = test_result_VarCoNetV2_aicha['test_result'][0][1]
AICHA_0_2 = test_result_VarCoNetV2_aicha['test_result'][0][2]
AICHA_1_1 = test_result_VarCoNetV2_aicha['test_result'][0][3]
AICHA_1_2 = test_result_VarCoNetV2_aicha['test_result'][0][4]
AICHA_2_2 = test_result_VarCoNetV2_aicha['test_result'][0][5]


plt.figure()
y = np.concatenate((AAL_0_0, AAL_0_1, AAL_0_2, AAL_1_1, AAL_1_2, AAL_2_2,
                    AICHA_0_0, AICHA_0_1, AICHA_0_2, AICHA_1_1, AICHA_1_2, AICHA_2_2), axis=0)
x = np.concatenate((np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),
                    np.full((10,), '2-2'),np.full((10,), '2-5'),
                    np.full((10,), '2-8'),np.full((10,), '5-5'),
                    np.full((10,), '5-8'),np.full((10,), '8-8'),),axis=0)
hue = np.concatenate((np.full((60,), 'AAL'), np.full((60,), 'AICHA')),axis=0)

d = {'Fingerprinting Accuracy': y, 'Atlas': hue, 'Duration Combinations (mins)': x}
df = pd.DataFrame(data=d)
sns.barplot(df, x="Duration Combinations (mins)", y="Fingerprinting Accuracy", hue="Atlas")
plt.ylabel('Fingerprinting Accuracy', fontsize=17)
plt.xlabel('Duration Combinations (mins)', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize = 16)
plt.tight_layout()
plt.savefig(os.path.join("ablations","plots","HCP_atlas.png"), dpi=600, bbox_inches='tight')
plt.close()

