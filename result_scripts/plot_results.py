import pickle
import numpy as np
import os  
from sklearn.metrics import f1_score,roc_curve
import pandas as pd
import matplotlib.pyplot as plt

path_results = r'/media/student1/ExtremePro/VarCoNet_results_revision' 

if not os.path.exists(os.path.join(path_results, 'gathered_results.csv')):
    rows = []
    atlases = ['AAL', 'AICHA']
    for atlas in atlases:   
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_A_GCL_results.pkl'), 'rb') as f:
            test_result_A_GCL = pickle.load(f) 
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_BNT_results.pkl'), 'rb') as f:
            test_result_BNT = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_BolT_results.pkl'), 'rb') as f:
            test_result_BolT = pickle.load(f) 
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_BrainIB_results.pkl'), 'rb') as f:
            test_result_BrainIB = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_CVFormer_results.pkl'), 'rb') as f:
            test_result_CVFormer = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_DeepFMRI_results.pkl'), 'rb') as f:
            test_result_DeepFMRI = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_FBNETGEN_results.pkl'), 'rb') as f:
            test_result_FBNETGEN = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_GCDA_results.pkl'), 'rb') as f:
            test_result_GCDA = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_UCGL_results.pkl'), 'rb') as f:
            test_result_UCGL = pickle.load(f)
        with open(os.path.join(path_results,'results_' + 'ABIDEI', atlas, 'ABIDEI' + '_VarCoNet_results.pkl'), 'rb') as f:
            test_result_VarCoNet = pickle.load(f)
        if atlas == 'AAL':
            with open(os.path.join(path_results,'results_' + 'ABIDEI', 'ABIDEI' + '_BAnD_results.pkl'), 'rb') as f:
                test_result_BAnD = pickle.load(f)
            with open(os.path.join(path_results,'results_' + 'ABIDEI', 'ABIDEI' + '_LFB_results.pkl'), 'rb') as f:
                test_result_LFB = pickle.load(f)
           
        '''-----------------------------------------BolT--------------------------------------------------'''
        val_losses = np.array(test_result_BolT['val_losses'])  
        val_aucs = np.array(test_result_BolT['val_aucs']) 
        val_probs = test_result_BolT['val_probs']
        y_vals = test_result_BolT['y_val']
        test_losses = np.array(test_result_BolT['test_losses'])    
        test_aucs = np.array(test_result_BolT['test_aucs'])
        test_probs = test_result_BolT['test_probs']
        y_tests = test_result_BolT['y_test']
        val_losses_ext = np.array(test_result_BolT['val_losses_ext'])
        val_aucs_ext = np.array(test_result_BolT['val_aucs_ext'])
        val_probs_ext = test_result_BolT['val_probs_ext'] 
        y_vals_ext = test_result_BolT['y_val_ext']
        ext_test_losses = np.array(test_result_BolT['ext_test_losses'])
        ext_test_aucs = np.array(test_result_BolT['ext_test_aucs'])
        ext_test_probs = test_result_BolT['ext_test_probs']
        y_ext_test = test_result_BolT['y_ext_test']
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)
        BolT_val_f1, BolT_test_f1, BolT_val_f1_ext, BolT_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
        for i in range(len(val_probs)):
            val_prob = val_probs[i][min_indices[i]]
            y_val = y_vals[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            BolT_val_f1[i] = f1_score(y_val, val_prob)
            test_prob = test_probs[i][min_indices[i]]
            y_test = y_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            BolT_test_f1[i] = f1_score(y_test, test_prob)
        for i in range(len(val_probs_ext)):
            val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
            y_val_ext = y_vals_ext[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            BolT_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            BolT_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

        BolT_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        BolT_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze()
        BolT_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        BolT_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        
        BolT_test_aucs = BolT_test_aucs*100
        BolT_ext_test_aucs = BolT_ext_test_aucs*100
        BolT_test_f1 = BolT_test_f1*100
        BolT_ext_test_f1 = BolT_ext_test_f1*100
        
        '''-----------------------------------------BNT--------------------------------------------------'''
        val_losses = np.array(test_result_BNT['val_losses'])  
        val_aucs = np.array(test_result_BNT['val_aucs']) 
        val_probs = test_result_BNT['val_probs']
        y_vals = test_result_BNT['y_val']
        test_losses = np.array(test_result_BNT['test_losses'])    
        test_aucs = np.array(test_result_BNT['test_aucs'])
        test_probs = test_result_BNT['test_probs']
        y_tests = test_result_BNT['y_test']
        val_losses_ext = np.array(test_result_BNT['val_losses_ext'])
        val_aucs_ext = np.array(test_result_BNT['val_aucs_ext'])
        val_probs_ext = test_result_BNT['val_probs_ext'] 
        y_vals_ext = test_result_BNT['y_val_ext']
        ext_test_losses = np.array(test_result_BNT['ext_test_losses'])
        ext_test_aucs = np.array(test_result_BNT['ext_test_aucs'])
        ext_test_probs = test_result_BNT['ext_test_probs']
        y_ext_tests = test_result_BNT['y_ext_test']
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)
        BNT_val_f1, BNT_test_f1, BNT_val_f1_ext, BNT_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
        for i in range(len(val_probs)):
            val_prob = val_probs[i][min_indices[i]]
            y_val = y_vals[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            BNT_val_f1[i] = f1_score(y_val, val_prob)
            test_prob = test_probs[i][min_indices[i]]
            y_test = y_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            BNT_test_f1[i] = f1_score(y_test, test_prob)
        for i in range(len(val_probs_ext)):
            val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
            y_val_ext = y_vals_ext[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            BNT_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
            y_ext_test = y_ext_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            BNT_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

        BNT_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        BNT_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze()
        BNT_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        BNT_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        
        BNT_test_aucs = BNT_test_aucs*100
        BNT_ext_test_aucs = BNT_ext_test_aucs*100
        BNT_test_f1 = BNT_test_f1*100
        BNT_ext_test_f1 = BNT_ext_test_f1*100
        

        '''-----------------------------------------CVFormer--------------------------------------------------'''
        val_losses = np.array(test_result_CVFormer['val_losses'])  
        val_aucs = np.array(test_result_CVFormer['val_aucs']) 
        val_probs = test_result_CVFormer['val_probs']
        y_vals = test_result_CVFormer['y_val']
        test_losses = np.array(test_result_CVFormer['test_losses'])    
        test_aucs = np.array(test_result_CVFormer['test_aucs'])
        test_probs = test_result_CVFormer['test_probs']
        y_tests = test_result_CVFormer['y_test']
        val_losses_ext = np.array(test_result_CVFormer['val_losses_ext'])
        val_aucs_ext = np.array(test_result_CVFormer['val_aucs_ext'])
        val_probs_ext = test_result_CVFormer['val_probs_ext'] 
        y_vals_ext = test_result_CVFormer['y_val_ext']
        ext_test_losses = np.array(test_result_CVFormer['ext_test_losses'])
        ext_test_aucs = np.array(test_result_CVFormer['ext_test_aucs'])
        ext_test_probs = test_result_CVFormer['ext_test_probs']
        y_ext_test = test_result_CVFormer['y_ext_test']
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)

        CVFormer_val_f1, CVFormer_test_f1, CVFormer_val_f1_ext, CVFormer_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
        for i in range(len(val_probs)):
            val_prob = val_probs[i][min_indices[i]]
            y_val = y_vals[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            CVFormer_val_f1[i] = f1_score(y_val, val_prob)
            test_prob = test_probs[i][min_indices[i]]
            y_test = y_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            CVFormer_test_f1[i] = f1_score(y_test, test_prob)
        for i in range(len(val_probs_ext)):
            val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
            y_val_ext = y_vals_ext[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            CVFormer_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            CVFormer_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

        CVFormer_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        CVFormer_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze()
        CVFormer_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        CVFormer_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        
        CVFormer_test_aucs = CVFormer_test_aucs*100
        CVFormer_ext_test_aucs = CVFormer_ext_test_aucs*100
        CVFormer_test_f1 = CVFormer_test_f1*100
        CVFormer_ext_test_f1 = CVFormer_ext_test_f1*100

        '''-----------------------------------------FBNETGEN--------------------------------------------------'''
        val_losses = np.array(test_result_FBNETGEN['val_losses'])  
        val_aucs = np.array(test_result_FBNETGEN['val_aucs']) 
        val_probs = test_result_FBNETGEN['val_probs']
        y_vals = test_result_FBNETGEN['y_val']
        test_losses = np.array(test_result_FBNETGEN['test_losses'])    
        test_aucs = np.array(test_result_FBNETGEN['test_aucs'])
        test_probs = test_result_FBNETGEN['test_probs']
        y_tests = test_result_FBNETGEN['y_test']
        val_losses_ext = np.array(test_result_FBNETGEN['val_losses_ext'])
        val_aucs_ext = np.array(test_result_FBNETGEN['val_aucs_ext'])
        val_probs_ext = test_result_FBNETGEN['val_probs_ext'] 
        y_vals_ext = test_result_FBNETGEN['y_val_ext']
        ext_test_losses = np.array(test_result_FBNETGEN['ext_test_losses'])
        ext_test_aucs = np.array(test_result_FBNETGEN['ext_test_aucs'])
        ext_test_probs = test_result_FBNETGEN['ext_test_probs']
        y_ext_tests = test_result_FBNETGEN['y_ext_test']
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)

        FBNETGEN_val_f1, FBNETGEN_test_f1, FBNETGEN_val_f1_ext, FBNETGEN_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)),np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
        for i in range(len(val_probs)):
            val_prob = val_probs[i][min_indices[i]]
            y_val = y_vals[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            FBNETGEN_val_f1[i] = f1_score(y_val, val_prob)
            test_prob = test_probs[i][min_indices[i]]
            y_test = y_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            FBNETGEN_test_f1[i] = f1_score(y_test, test_prob)
        for i in range(len(val_probs_ext)):
            val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
            y_val_ext = y_vals_ext[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            FBNETGEN_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
            y_ext_test = y_ext_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            FBNETGEN_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)
 
        FBNETGEN_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        FBNETGEN_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        FBNETGEN_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        FBNETGEN_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        
        FBNETGEN_test_aucs = FBNETGEN_test_aucs*100
        FBNETGEN_ext_test_aucs = FBNETGEN_ext_test_aucs*100
        FBNETGEN_test_f1 = FBNETGEN_test_f1*100
        FBNETGEN_ext_test_f1 = FBNETGEN_ext_test_f1*100

        '''-----------------------------------------DeepFMRI--------------------------------------------------'''
        val_losses = np.array(test_result_DeepFMRI['val_losses'])  
        val_aucs = np.array(test_result_DeepFMRI['val_aucs']) 
        val_probs = test_result_DeepFMRI['val_probs']
        y_vals = test_result_DeepFMRI['y_val']
        test_losses = np.array(test_result_DeepFMRI['test_losses'])    
        test_aucs = np.array(test_result_DeepFMRI['test_aucs'])
        test_probs = test_result_DeepFMRI['test_probs']
        y_tests = test_result_DeepFMRI['y_test']
        val_losses_ext = np.array(test_result_DeepFMRI['val_losses_ext'])
        val_aucs_ext = np.array(test_result_DeepFMRI['val_aucs_ext'])
        val_probs_ext = test_result_DeepFMRI['val_probs_ext'] 
        y_vals_ext = test_result_DeepFMRI['y_val_ext']
        ext_test_losses = np.array(test_result_DeepFMRI['ext_test_losses'])
        ext_test_aucs = np.array(test_result_DeepFMRI['ext_test_aucs'])
        ext_test_probs = test_result_DeepFMRI['ext_test_probs']
        y_ext_test = test_result_DeepFMRI['y_ext_test']
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)

        DeepFMRI_val_f1, DeepFMRI_test_f1, DeepFMRI_val_f1_ext, DeepFMRI_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
        for i in range(len(val_probs)):
            val_prob = val_probs[i][min_indices[i]]
            y_val = y_vals[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            DeepFMRI_val_f1[i] = f1_score(y_val, val_prob)
            test_prob = test_probs[i][min_indices[i]]
            y_test = y_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            DeepFMRI_test_f1[i] = f1_score(y_test, test_prob)
        for i in range(len(val_probs_ext)):
            val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
            y_val_ext = y_vals_ext[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            DeepFMRI_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            DeepFMRI_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

        DeepFMRI_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        DeepFMRI_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        DeepFMRI_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        DeepFMRI_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        
        DeepFMRI_test_aucs = DeepFMRI_test_aucs*100
        DeepFMRI_ext_test_aucs = DeepFMRI_ext_test_aucs*100
        DeepFMRI_test_f1 = DeepFMRI_test_f1*100
        DeepFMRI_ext_test_f1 = DeepFMRI_ext_test_f1*100

        '''-----------------------------------------BrainIB--------------------------------------------------'''
        val_losses = np.array(test_result_BrainIB['val_losses'])  
        val_aucs = np.array(test_result_BrainIB['val_aucs']) 
        val_probs = test_result_BrainIB['val_probs']
        y_vals = test_result_BrainIB['y_val']
        test_losses = np.array(test_result_BrainIB['test_losses'])    
        test_aucs = np.array(test_result_BrainIB['test_aucs'])
        test_probs = test_result_BrainIB['test_probs']
        y_tests = test_result_BrainIB['y_test']
        val_losses_ext = np.array(test_result_BrainIB['val_losses_ext'])
        val_aucs_ext = np.array(test_result_BrainIB['val_aucs_ext'])
        val_probs_ext = test_result_BrainIB['val_probs_ext'] 
        y_vals_ext = test_result_BrainIB['y_val_ext']
        ext_test_losses = np.array(test_result_BrainIB['ext_test_losses'])
        ext_test_aucs = np.array(test_result_BrainIB['ext_test_aucs'])
        ext_test_probs = test_result_BrainIB['ext_test_probs']
        y_ext_test = test_result_BrainIB['y_ext_test']
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)

        BrainIB_val_f1, BrainIB_test_f1, BrainIB_val_f1_ext, BrainIB_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)), np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
        for i in range(len(val_probs)):
            val_prob = val_probs[i][min_indices[i]]
            y_val = y_vals[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            BrainIB_val_f1[i] = f1_score(y_val, val_prob)
            test_prob = test_probs[i][min_indices[i]]
            y_test = y_tests[i][min_indices[i]]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            BrainIB_test_f1[i] = f1_score(y_test, test_prob)
        for i in range(len(val_probs_ext)):
            val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
            y_val_ext = y_vals_ext[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            BrainIB_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            BrainIB_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

        BrainIB_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        BrainIB_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        BrainIB_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        BrainIB_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        
        BrainIB_test_aucs = BrainIB_test_aucs*100
        BrainIB_ext_test_aucs = BrainIB_ext_test_aucs*100
        BrainIB_test_f1 = BrainIB_test_f1*100
        BrainIB_ext_test_f1 = BrainIB_ext_test_f1*100
        
        '''-----------------------------------------A-GCL--------------------------------------------------'''
        val_losses = np.zeros((100,100))   
        val_aucs = np.zeros((100,100))  
        val_f1 = np.zeros((100,100))  
        test_losses = np.zeros((100,100))   
        test_aucs = np.zeros((100,100))  
        test_f1 = np.zeros((100,100))  
        val_losses_ext = np.zeros((10,100))   
        val_aucs_ext = np.zeros((10,100))  
        val_f1_ext = np.zeros((10,100)) 
        ext_test_losses = np.zeros((10,100))   
        ext_test_aucs = np.zeros((10,100))  
        ext_test_f1 = np.zeros((10,100))
        for i,test in enumerate(test_result_A_GCL['epoch_results']):
            for j,test1 in enumerate(test):
                val_aucs[i,j] = test1['val_auc']
                val_losses[i,j] = test1['val_loss']
                val_prob = test1['val_probs']
                y_val = test1['y_val']
                fpr, tpr, thresholds = roc_curve(y_val, val_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                val_prob[val_prob<best_threshold] = 0
                val_prob[val_prob>=best_threshold] = 1
                val_f1[i,j] = f1_score(y_val, val_prob)
                test_aucs[i,j] = test1['test_auc']
                test_losses[i,j] = test1['test_loss']
                test_prob = test1['test_probs']
                y_test = test1['y_test']
                fpr, tpr, thresholds = roc_curve(y_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                test_prob[test_prob<best_threshold] = 0
                test_prob[test_prob>=best_threshold] = 1
                test_f1[i,j] = f1_score(y_test, test_prob)
        for i,test in enumerate(test_result_A_GCL['epoch_results_ext']):
            for j,test1 in enumerate(test):
                val_aucs_ext[i,j] = test1['val_auc']
                val_losses_ext[i,j] = test1['val_loss']
                val_prob_ext = test1['val_probs']
                y_val_ext = test1['y_val']
                fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                val_prob_ext[val_prob_ext<best_threshold] = 0
                val_prob_ext[val_prob_ext>=best_threshold] = 1
                val_f1_ext[i,j] = f1_score(y_val_ext, val_prob_ext)
                ext_test_aucs[i,j] = test1['test_auc']
                ext_test_losses[i,j] = test1['test_loss']
                ext_test_prob = test1['test_probs']
                y_ext_test = test1['y_test']
                fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                ext_test_prob[ext_test_prob<best_threshold] = 0
                ext_test_prob[ext_test_prob>=best_threshold] = 1
                ext_test_f1[i,j] = f1_score(y_ext_test, ext_test_prob)
                
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)
                
        A_GCL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        A_GCL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        A_GCL_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 
        A_GCL_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        A_GCL_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        A_GCL_ext_test_f1 = np.take_along_axis(ext_test_f1, min_indices_ext[:, None], axis=1).squeeze()
        
        A_GCL_test_aucs = A_GCL_test_aucs*100
        A_GCL_ext_test_aucs = A_GCL_ext_test_aucs*100
        A_GCL_test_f1 = A_GCL_test_f1*100
        A_GCL_ext_test_f1 = A_GCL_ext_test_f1*100
        
        '''-----------------------------------------UCGL--------------------------------------------------'''
        val_losses = np.zeros((100,30))   
        val_aucs = np.zeros((100,30))  
        val_f1 = np.zeros((100,))  
        test_losses = np.zeros((100,30))   
        test_aucs = np.zeros((100,30))  
        test_f1 = np.zeros((100,))  
        val_losses_ext = np.zeros((10,30))   
        val_aucs_ext = np.zeros((10,30))  
        val_f1_ext = np.zeros((10,)) 
        ext_test_losses = np.zeros((10,30))   
        ext_test_aucs = np.zeros((10,30))  
        ext_test_f1 = np.zeros((10,))
        for i,test in enumerate(test_result_UCGL['epoch_results']):
            val_aucs[i,:] = test['val_aucs']
            val_losses[i,:] = test['val_loss']
            val_prob = test['val_probs']
            y_val = test['y_val']
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            val_f1[i] = f1_score(y_val, val_prob)
            test_aucs[i,:] = test['test_auc']
            test_losses[i,:] = test['test_loss']
            test_prob = test['test_probs']
            y_test = test['y_test']
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            test_f1[i] = f1_score(y_test, test_prob)
        for i,test in enumerate(test_result_UCGL['epoch_results_ext']):
            val_aucs_ext[i,:] = test['val_aucs']
            val_losses_ext[i,:] = test['val_loss']
            val_prob_ext = test['val_probs']
            y_val_ext = test['y_val']
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_aucs[i,:] = test['test_auc']
            ext_test_losses[i,:] = test['test_loss']
            ext_test_prob = test['test_probs']
            y_ext_test = test['y_test']
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)
                
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)
                
        UCGL_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        UCGL_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        UCGL_test_f1 = test_f1
        UCGL_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        UCGL_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        UCGL_ext_test_f1 = ext_test_f1
        
        UCGL_test_aucs = UCGL_test_aucs*100
        UCGL_ext_test_aucs = UCGL_ext_test_aucs*100
        UCGL_test_f1 = UCGL_test_f1*100
        UCGL_ext_test_f1 = UCGL_ext_test_f1*100

        '''-----------------------------------------GCDA--------------------------------------------------'''
        val_losses = np.zeros((100,30))   
        val_aucs = np.zeros((100,30))  
        val_f1 = np.zeros((100,))  
        test_losses = np.zeros((100,30))   
        test_aucs = np.zeros((100,30))  
        test_f1 = np.zeros((100,))  
        val_losses_ext = np.zeros((10,30))   
        val_aucs_ext = np.zeros((10,30))  
        val_f1_ext = np.zeros((10,)) 
        ext_test_losses = np.zeros((10,30))   
        ext_test_aucs = np.zeros((10,30))  
        ext_test_f1 = np.zeros((10,))
        for i,test in enumerate(test_result_GCDA['epoch_results']):
            val_aucs[i,:] = test['val_aucs']
            val_losses[i,:] = test['val_loss']
            val_prob = test['val_probs'][0]
            y_val = test['y_val']
            fpr, tpr, thresholds = roc_curve(y_val, val_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob[val_prob<best_threshold] = 0
            val_prob[val_prob>=best_threshold] = 1
            val_f1[i] = f1_score(y_val, val_prob)
            test_aucs[i,:] = test['test_auc']
            test_losses[i,:] = test['test_loss']
            test_prob = test['test_probs'][0]
            y_test = test['y_test']
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            test_f1[i] = f1_score(y_test, test_prob)
        for i,test in enumerate(test_result_GCDA['epoch_results_ext']):
            val_aucs_ext[i,:] = test['val_aucs']
            val_losses_ext[i,:] = test['val_loss']
            val_prob_ext = test['val_probs'][0]
            y_val_ext = test['y_val']
            fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            val_prob_ext[val_prob_ext<best_threshold] = 0
            val_prob_ext[val_prob_ext>=best_threshold] = 1
            val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
            ext_test_aucs[i,:] = test['test_auc']
            ext_test_losses[i,:] = test['test_loss']
            ext_test_prob = test['test_probs'][0]
            y_ext_test = test['y_test']
            fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            ext_test_prob[ext_test_prob<best_threshold] = 0
            ext_test_prob[ext_test_prob>=best_threshold] = 1
            ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)
                
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)
                
        GCDA_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        GCDA_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        GCDA_test_f1 = test_f1
        GCDA_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        GCDA_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        GCDA_ext_test_f1 = ext_test_f1
        
        GCDA_test_aucs = GCDA_test_aucs*100
        GCDA_ext_test_aucs = GCDA_ext_test_aucs*100
        GCDA_test_f1 = GCDA_test_f1*100
        GCDA_ext_test_f1 = GCDA_ext_test_f1*100
        
        '''-----------------------------------------VarCoNet--------------------------------------------------'''
        val_losses = np.zeros((100,50))   
        val_aucs = np.zeros((100,50))  
        val_f1 = np.zeros((100,50))  
        test_losses = np.zeros((100,50))   
        test_aucs = np.zeros((100,50))  
        test_f1 = np.zeros((100,50))  
        val_losses_ext = np.zeros((10,50))   
        val_aucs_ext = np.zeros((10,50))  
        val_f1_ext = np.zeros((10,50)) 
        ext_test_losses = np.zeros((10,50))   
        ext_test_aucs = np.zeros((10,50))  
        ext_test_f1 = np.zeros((10,50))
        for i,test in enumerate(test_result_VarCoNet['epoch_results']):
            for j,test1 in enumerate(test):
                val_aucs[i,j] = test1['best_val_auc']
                val_losses[i,j] = test1['best_val_loss']
                val_prob = test1['val_probs']
                y_val = test1['y_val']
                fpr, tpr, thresholds = roc_curve(y_val, val_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                val_prob[val_prob<best_threshold] = 0
                val_prob[val_prob>=best_threshold] = 1
                val_f1[i,j] = f1_score(y_val, val_prob)
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
        for i,test in enumerate(test_result_VarCoNet['epoch_results_ext']):
            for j,test1 in enumerate(test):
                val_aucs_ext[i,j] = test1['best_val_auc']
                val_losses_ext[i,j] = test1['best_val_loss']
                val_prob_ext = test1['val_probs']
                y_val_ext = test1['y_val']
                fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                val_prob_ext[val_prob_ext<best_threshold] = 0
                val_prob_ext[val_prob_ext>=best_threshold] = 1
                val_f1_ext[i,j] = f1_score(y_val_ext, val_prob_ext)
                ext_test_aucs[i,j] = test1['best_test_auc']
                ext_test_losses[i,j] = test1['best_test_loss']
                ext_test_prob = test1['test_probs']
                y_ext_test = test1['y_test']
                fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                ext_test_prob[ext_test_prob<best_threshold] = 0
                ext_test_prob[ext_test_prob>=best_threshold] = 1
                ext_test_f1[i,j] = f1_score(y_ext_test, ext_test_prob)
                
        min_indices = np.argmin(val_losses, axis=1)
        min_indices_ext = np.argmin(val_losses_ext, axis=1)
                
        VarCoNet_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
        VarCoNet_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
        VarCoNet_test_f1 = np.take_along_axis(test_f1, min_indices[:, None], axis=1).squeeze() 
        VarCoNet_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
        VarCoNet_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()
        VarCoNet_ext_test_f1 = np.take_along_axis(ext_test_f1, min_indices_ext[:, None], axis=1).squeeze()

        VarCoNet_test_aucs = VarCoNet_test_aucs*100
        VarCoNet_ext_test_aucs = VarCoNet_ext_test_aucs*100
        VarCoNet_test_f1 = VarCoNet_test_f1*100
        VarCoNet_ext_test_f1 = VarCoNet_ext_test_f1*100
        
        if atlas == 'AAL':
            '''-----------------------------------------BAnD-------------------------------------------------'''
            val_losses = np.array(test_result_BAnD['val_losses'])  
            val_aucs = np.array(test_result_BAnD['val_aucs']) 
            val_probs = test_result_BAnD['val_probs']
            y_vals = test_result_BAnD['y_val']
            test_losses = np.array(test_result_BAnD['test_losses'])    
            test_aucs = np.array(test_result_BAnD['test_aucs'])
            test_probs = test_result_BAnD['test_probs']
            y_tests = test_result_BAnD['y_test']
            val_losses_ext = np.array(test_result_BAnD['val_losses_ext'])
            val_aucs_ext = np.array(test_result_BAnD['val_aucs_ext'])
            val_probs_ext = test_result_BAnD['val_probs_ext'] 
            y_vals_ext = test_result_BAnD['y_val_ext']
            ext_test_losses = np.array(test_result_BAnD['ext_test_losses'])
            ext_test_aucs = np.array(test_result_BAnD['ext_test_aucs'])
            ext_test_probs = test_result_BAnD['ext_test_probs']
            y_ext_test = test_result_BAnD['y_ext_test']
            min_indices = np.argmin(val_losses, axis=1)
            min_indices_ext = np.argmin(val_losses_ext, axis=1)

            BAnD_val_f1, BAnD_test_f1, BAnD_val_f1_ext, BAnD_ext_test_f1 = np.zeros((len(val_probs),)), np.zeros((len(val_probs),)),np.zeros((len(val_probs_ext),)), np.zeros((len(val_probs_ext),))
            for i in range(len(val_probs)):
                val_prob = val_probs[i][min_indices[i]]
                y_val = y_vals[i][min_indices[i]]
                fpr, tpr, thresholds = roc_curve(y_val, val_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                val_prob[val_prob<best_threshold] = 0
                val_prob[val_prob>=best_threshold] = 1
                BAnD_val_f1[i] = f1_score(y_val, val_prob)
                test_prob = test_probs[i][min_indices[i]]
                y_test = y_tests[i][min_indices[i]]
                fpr, tpr, thresholds = roc_curve(y_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                test_prob[test_prob<best_threshold] = 0
                test_prob[test_prob>=best_threshold] = 1
                BAnD_test_f1[i] = f1_score(y_test, test_prob)
            for i in range(len(val_probs_ext)):
                val_prob_ext = val_probs_ext[i][min_indices_ext[i]]
                y_val_ext = y_vals_ext[i][min_indices_ext[i]]
                fpr, tpr, thresholds = roc_curve(y_val_ext, val_prob_ext)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                val_prob_ext[val_prob_ext<best_threshold] = 0
                val_prob_ext[val_prob_ext>=best_threshold] = 1
                BAnD_val_f1_ext[i] = f1_score(y_val_ext, val_prob_ext)
                ext_test_prob = ext_test_probs[i][min_indices_ext[i]]
                fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                ext_test_prob[ext_test_prob<best_threshold] = 0
                ext_test_prob[ext_test_prob>=best_threshold] = 1
                BAnD_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)

            BAnD_test_losses = np.take_along_axis(test_losses, min_indices[:, None], axis=1).squeeze()
            BAnD_test_aucs = np.take_along_axis(test_aucs, min_indices[:, None], axis=1).squeeze() 
            BAnD_ext_test_losses = np.take_along_axis(ext_test_losses, min_indices_ext[:, None], axis=1).squeeze()
            BAnD_ext_test_aucs = np.take_along_axis(ext_test_aucs, min_indices_ext[:, None], axis=1).squeeze()

            BAnD_test_aucs = BAnD_test_aucs*100
            BAnD_ext_test_aucs = BAnD_ext_test_aucs*100
            BAnD_test_f1 = BAnD_test_f1*100
            BAnD_ext_test_f1 = BAnD_ext_test_f1*100

            '''-----------------------------------------LFB--------------------------------------------------'''
            LFB_val_losses = np.array(test_result_LFB['val_losses'])  
            LFB_test_losses = np.array(test_result_LFB['test_losses'])    
            LFB_test_aucs = np.array(test_result_LFB['test_aucs'])
            test_probs = test_result_LFB['test_probs']
            y_tests = test_result_LFB['y_test']
            LFB_val_losses_ext = np.array(test_result_LFB['val_losses_ext'])
            LFB_ext_test_losses = np.array(test_result_LFB['ext_test_losses'])
            LFB_ext_test_aucs = np.array(test_result_LFB['ext_test_aucs'])
            ext_test_probs = test_result_LFB['ext_test_probs']
            y_ext_test = test_result_LFB['y_ext_test']

            LFB_test_f1, LFB_ext_test_f1 = np.zeros((len(val_losses),)), np.zeros((len(val_losses_ext),))
            for i in range(len(val_losses)):
                test_prob = test_probs[i]
                y_test = y_tests[i]
                fpr, tpr, thresholds = roc_curve(y_test, test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                test_prob[test_prob<best_threshold] = 0
                test_prob[test_prob>=best_threshold] = 1
                LFB_test_f1[i] = f1_score(y_test, test_prob)
            for i in range(len(val_losses_ext)):
                ext_test_prob = ext_test_probs[i]
                fpr, tpr, thresholds = roc_curve(y_ext_test, ext_test_prob)
                youden_j = tpr - fpr
                best_index = np.argmax(youden_j)
                best_threshold = thresholds[best_index]
                ext_test_prob[ext_test_prob<best_threshold] = 0
                ext_test_prob[ext_test_prob>=best_threshold] = 1
                LFB_ext_test_f1[i] = f1_score(y_ext_test, ext_test_prob)
            
            LFB_test_aucs = LFB_test_aucs*100
            LFB_ext_test_aucs = LFB_ext_test_aucs*100
            LFB_test_f1 = LFB_test_f1*100
            LFB_ext_test_f1 = LFB_ext_test_f1*100
                
        for s in range(100):
            bce = A_GCL_test_losses[s]
            auc = A_GCL_test_aucs[s]
            f1  = A_GCL_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'A-GCL', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'A-GCL', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'A-GCL', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = BNT_test_losses[s]
            auc = BNT_test_aucs[s]
            f1  = BNT_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'BNT', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'BNT', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'BNT', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = BolT_test_losses[s]
            auc = BolT_test_aucs[s]
            f1  = BolT_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'BolT', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'BolT', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'BolT', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = BrainIB_test_losses[s]
            auc = BrainIB_test_aucs[s]
            f1  = BrainIB_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'BrainIB', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'BrainIB', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'BrainIB', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = CVFormer_test_losses[s]
            auc = CVFormer_test_aucs[s]
            f1  = CVFormer_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'CVFormer', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'CVFormer', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'CVFormer', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = DeepFMRI_test_losses[s]
            auc = DeepFMRI_test_aucs[s]
            f1  = DeepFMRI_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'DeepFMRI', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'DeepFMRI', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'DeepFMRI', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = FBNETGEN_test_losses[s]
            auc = FBNETGEN_test_aucs[s]
            f1  = FBNETGEN_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'FBNETGEN', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'FBNETGEN', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'FBNETGEN', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = UCGL_test_losses[s]
            auc = UCGL_test_aucs[s]
            f1  = UCGL_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'UCGL', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'UCGL', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'UCGL', "F1 (%)",  float(f1), s),
            ])
        for s in range(100):
            bce = GCDA_test_losses[s]
            auc = GCDA_test_aucs[s]
            f1  = GCDA_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'GCDA', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'GCDA', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'GCDA', "F1 (%)",  float(f1), s),
            ])
        if atlas == 'AAL':
            for s in range(10):
                bce = LFB_test_losses[s]
                auc = LFB_test_aucs[s]
                f1  = LFB_test_f1[s]
                rows.extend([
                    ('10-Fold CV', 'DiFuMo', 'CSM', "BCE", float(np.clip(bce, 0, 1)), s),
                    ('10-Fold CV', 'DiFuMo', 'CSM', "AUC (%)", float(auc), s),
                    ('10-Fold CV', 'DiFuMo', 'CSM', "F1 (%)",  float(f1), s),
                ])
            for s in range(10):
                bce = BAnD_test_losses[s]
                auc = BAnD_test_aucs[s]
                f1  = BAnD_test_f1[s]
                rows.extend([
                    ('10-Fold CV', 'Volumetric', 'BAnD', "BCE", float(np.clip(bce, 0, 1)), s),
                    ('10-Fold CV', 'Volumetric', 'BAnD', "AUC (%)", float(auc), s),
                    ('10-Fold CV', 'Volumetric', 'BAnD', "F1 (%)",  float(f1), s),
                ])
        for s in range(100):
            bce = VarCoNet_test_losses[s]
            auc = VarCoNet_test_aucs[s]
            f1  = VarCoNet_test_f1[s]
            rows.extend([
                ('10-Fold CV', atlas, 'VarCoNet', "BCE", float(np.clip(bce, 0, 1)), s),
                ('10-Fold CV', atlas, 'VarCoNet', "AUC (%)", float(auc), s),
                ('10-Fold CV', atlas, 'VarCoNet', "F1 (%)",  float(f1), s),
            ])
            
        for s in range(10):
            bce = A_GCL_ext_test_losses[s]
            auc = A_GCL_ext_test_aucs[s]
            f1  = A_GCL_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'A-GCL', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'A-GCL', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'A-GCL', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = BNT_ext_test_losses[s]
            auc = BNT_ext_test_aucs[s]
            f1  = BNT_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'BNT', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'BNT', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'BNT', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = BolT_ext_test_losses[s]
            auc = BolT_ext_test_aucs[s]
            f1  = BolT_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'BolT', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'BolT', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'BolT', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = BrainIB_ext_test_losses[s]
            auc = BrainIB_ext_test_aucs[s]
            f1  = BrainIB_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'BrainIB', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'BrainIB', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'BrainIB', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = CVFormer_ext_test_losses[s]
            auc = CVFormer_ext_test_aucs[s]
            f1  = CVFormer_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'CVFormer', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'CVFormer', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'CVFormer', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = DeepFMRI_ext_test_losses[s]
            auc = DeepFMRI_ext_test_aucs[s]
            f1  = DeepFMRI_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'DeepFMRI', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'DeepFMRI', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'DeepFMRI', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = FBNETGEN_ext_test_losses[s]
            auc = FBNETGEN_ext_test_aucs[s]
            f1  = FBNETGEN_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'FBNETGEN', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'FBNETGEN', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'FBNETGEN', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = UCGL_ext_test_losses[s]
            auc = UCGL_ext_test_aucs[s]
            f1  = UCGL_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'UCGL', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'UCGL', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'UCGL', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = GCDA_ext_test_losses[s]
            auc = GCDA_ext_test_aucs[s]
            f1  = GCDA_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'GCDA', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'GCDA', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'GCDA', "F1 (%)",  float(f1), s),
            ])
        if atlas == 'AAL':
            for s in range(10):
                bce = LFB_ext_test_losses[s]
                auc = LFB_ext_test_aucs[s]
                f1  = LFB_ext_test_f1[s]
                rows.extend([
                    ('External Test (Caltech)', 'DiFuMo', 'CSM', "BCE", float(np.clip(bce, 0, 1)), s),
                    ('External Test (Caltech)', 'DiFuMo', 'CSM', "AUC (%)", float(auc), s),
                    ('External Test (Caltech)', 'DiFuMo', 'CSM', "F1 (%)",  float(f1), s),
                ])
            for s in range(10):
                bce = BAnD_ext_test_losses[s]
                auc = BAnD_ext_test_aucs[s]
                f1  = BAnD_ext_test_f1[s]
                rows.extend([
                    ('External Test (Caltech)', 'Volumetric', 'BAnD', "BCE", float(np.clip(bce, 0, 1)), s),
                    ('External Test (Caltech)', 'Volumetric', 'BAnD', "AUC (%)", float(auc), s),
                    ('External Test (Caltech)', 'Volumetric', 'BAnD', "F1 (%)",  float(f1), s),
                ])
        for s in range(10):
            bce = VarCoNet_ext_test_losses[s]
            auc = VarCoNet_ext_test_aucs[s]
            f1  = VarCoNet_ext_test_f1[s]
            rows.extend([
                ('External Test (Caltech)', atlas, 'VarCoNet', "BCE", float(np.clip(bce, 0, 1)), s),
                ('External Test (Caltech)', atlas, 'VarCoNet', "AUC (%)", float(auc), s),
                ('External Test (Caltech)', atlas, 'VarCoNet', "F1 (%)",  float(f1), s),
            ])
        
        with open(os.path.join(path_results,'results_ABIDEII',atlas,'ABIDEII_results.pkl'), 'rb') as f:
            test_result_ABIDEII = pickle.load(f)
        test_result_BolT_ext = test_result_ABIDEII['BolT']
        test_result_VarCoNet_ext = test_result_ABIDEII['VarCoNet']
        
        BolT_test_losses = np.array(test_result_BolT_ext['test_losses'])
        BolT_test_aucs = np.array(test_result_BolT_ext['test_aucs'])
        test_probs = test_result_BolT_ext['test_probs']
        y_test = test_result_BolT_ext['y_test']
        BolT_test_f1 = np.zeros((len(BolT_test_losses),))
        for i in range(len(BolT_test_losses)):
            test_prob = test_probs[i]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            BolT_test_f1[i] = f1_score(y_test, test_prob)
            
        VarCoNet_test_losses = np.array(test_result_VarCoNet_ext['test_losses'])
        VarCoNet_test_aucs = np.array(test_result_VarCoNet_ext['test_aucs'])
        test_probs = test_result_VarCoNet_ext['test_probs']
        y_test = test_result_VarCoNet_ext['y_test']
        VarCoNet_test_f1 = np.zeros((len(VarCoNet_test_losses),))
        for i in range(len(VarCoNet_test_losses)):
            test_prob = test_probs[i]
            fpr, tpr, thresholds = roc_curve(y_test, test_prob)
            youden_j = tpr - fpr
            best_index = np.argmax(youden_j)
            best_threshold = thresholds[best_index]
            test_prob[test_prob<best_threshold] = 0
            test_prob[test_prob>=best_threshold] = 1
            VarCoNet_test_f1[i] = f1_score(y_test, test_prob)
        
        
        BolT_test_aucs = BolT_test_aucs*100
        BolT_test_f1 = BolT_test_f1*100
        VarCoNet_test_aucs = VarCoNet_test_aucs*100
        VarCoNet_test_f1 = VarCoNet_test_f1*100
            
        for s in range(10):
            bce = BolT_test_losses[s]
            auc = BolT_test_aucs[s]
            f1  = BolT_test_f1[s]
            rows.extend([
                ('ABIDEII', atlas, 'BolT', "BCE", float(np.clip(bce, 0, 1)), s),
                ('ABIDEII', atlas, 'BolT', "AUC (%)", float(auc), s),
                ('ABIDEII', atlas, 'BolT', "F1 (%)",  float(f1), s),
            ])
        for s in range(10):
            bce = VarCoNet_test_losses[s]
            auc = VarCoNet_test_aucs[s]
            f1  = VarCoNet_test_f1[s]
            rows.extend([
                ('ABIDEII', atlas, 'VarCoNet', "BCE", float(np.clip(bce, 0, 1)), s),
                ('ABIDEII', atlas, 'VarCoNet', "AUC (%)", float(auc), s),
                ('ABIDEII', atlas, 'VarCoNet', "F1 (%)",  float(f1), s),
            ])
    df = pd.DataFrame(rows, columns=['dataset', 'atlas', 'method', 'metric', 'value', 'seed'])
    df.to_csv(os.path.join(path_results, 'gathered_results.csv'))
else:
    df = pd.read_csv(os.path.join(path_results, 'gathered_results.csv'))

from benchmark_plots import (
    plot_metric_heatmap, plot_grouped_dotplot
)

ticks_font = 16
labels_font = 17
title_font = 18
legend_font = 12

fig = plot_grouped_dotplot(df, dataset="10-Fold CV", metric="AUC (%)", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, include_legend=True, legend_x=0.02, legend_y=0.96, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEI_kfold_AUC_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="10-Fold CV", metric="F1 (%)", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=0.5, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEI_kfold_F1_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="10-Fold CV", metric="BCE", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=1.0, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEI_kfold_BCE_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="External Test (Caltech)", metric="AUC (%)", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=0.5, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEI_ext_AUC_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="External Test (Caltech)", metric="F1 (%)", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=0.5, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEI_ext_F1_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="External Test (Caltech)", metric="BCE", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=1.0, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEI_ext_BCE_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="ABIDEII", metric="AUC (%)", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=0.5, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEII_AUC_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="ABIDEII", metric="F1 (%)", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=0.5, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEII_F1_results.png'), dpi=600, bbox_inches='tight')
plt.show()

fig = plot_grouped_dotplot(df, dataset="ABIDEII", metric="BCE", ticks_font=ticks_font, labels_font=labels_font, legend_font=legend_font, legend_x=0.86, legend_y=1.0, title_font=title_font)
fig.savefig(os.path.join(path_results,'plots','ABIDEII_BCE_results.png'), dpi=600, bbox_inches='tight')
plt.show()
