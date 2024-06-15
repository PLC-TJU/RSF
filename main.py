# This code is used for the paper "Enhancing Motor Imagery EEG Classification with a Riemannian Geometry-Based Spatial Filtering (RSF) Method"
# Author: Pan.LC <coreylin2023@outlook.com>
# Date: 2024/3/18
# License: MIT License

import os, time
import json
import numpy as np
import itertools
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import RepeatedStratifiedKFold
from contextlib import redirect_stdout, redirect_stderr

from moabb.paradigms import LeftRightImagery
from moabb.datasets import (Cho2017,
                            Weibo2014,
                            PhysionetMI,
                            Lee2019_MI,
                            Shin2017A,)
from loaddata.pan2023 import Pan2023
from loaddata import Dataset_Left_Right_MI
from deep_learning.dl_classifier import DL_Classifier
from rsf import RSF

def calc_acc(irun, traindata, trainlabel, testdata, testlabel, verbose = True, device = 'cuda'):
    method = irun[1]
    dim = irun[2]
    
    if method.lower() == 'none' and dim > 2:
        return [float('nan')] * len(algorithms), [float('nan')] * len(algorithms)
    
    Acc, Timecost = [], []

    # RSF
    start_time = time.monotonic()
    rsf_transformer = RSF(dim, method)
    traindata = rsf_transformer.fit_transform(traindata, trainlabel)
    testdata = rsf_transformer.transform(testdata)
    transform_time = time.monotonic() - start_time
    
    for model_name in algorithms:
        
        if not verbose:
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                testacc, elapsed_time = train_and_evaluate(traindata, trainlabel, testdata, testlabel, 
                                                           model_name, device, irun=irun)
        else:
            testacc, elapsed_time = train_and_evaluate(traindata, trainlabel, testdata, testlabel, 
                                                       model_name, device, irun=irun)
        
        Acc.append(testacc)
        Timecost.append(elapsed_time + transform_time)
              
    return Acc, Timecost

def train_and_evaluate(traindata, trainlabel, testdata, testlabel, 
                       model_name, device, rsf_method='none', rsf_dim=2,
                       irun=[0, 'none', 2]):
    
    start_time = time.monotonic()
    estimator = DL_Classifier(model_name=model_name, fs=fs, batch_size=32, lr=1e-2, max_epochs=200,device=device)
    try:
        estimator.fit(traindata, trainlabel)
        testacc = estimator.score(testdata, testlabel)
        elapsed_time = time.monotonic() - start_time
    except Exception as e:
        testacc = float('nan')
        elapsed_time = float('nan')
        with open(os.path.join(floder_name, 'error_{}.txt'.format(dataset_name)), 'w') as f:
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
            f.write(f"Error in {model_name} with {rsf_method} and {rsf_dim} on run {irun[0]}: {e}\n")
    return testacc, elapsed_time

def save_result(irun, result, filename):
    result_dict = {'run': irun[0], 'method': irun[1], 'dim': irun[2], 'Acc': result[0], 'Timecost': result[1]}
    try:
        with gpu_lock:
            with open(filename, 'a') as f:
                json.dump(result_dict, f)
                f.write('\n') 
    except:
        with open(filename, 'a') as f:
                json.dump(result_dict, f)
                f.write('\n') 

def check_completed_runs(filename, allrun):
    if not os.path.exists(filename):
        return allrun
    completed_runs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    results_start = lines.index("results:\n") + 1 if "results:\n" in lines else 0
    for line in lines[results_start:]:
        if line.strip():
            completed_runs.append(json.loads(line))
    completed_indices = [(run['run'], run['method'], run['dim']) for run in completed_runs]
    return [irun for irun in allrun if (irun[0], irun[1], irun[2]) not in completed_indices]
         
def split_data_for_cv(data, label):
    all_data = {}
    all_indices = {}
    for cv, (train_index, test_index) in enumerate(kf.split(data, label)):
        all_indices[cv] = {}
        all_indices[cv]['train_index'] = train_index
        all_indices[cv]['test_index'] = test_index
        all_data[cv] = {}
        all_data[cv]['traindata'] = data[train_index]
        all_data[cv]['trainlabel'] = label[train_index]
        all_data[cv]['testdata'] = data[test_index]
        all_data[cv]['testlabel'] = label[test_index]  
    return all_data, all_indices

def main_processes(dataset, personID, allrun):
    
    all_tasks = []
    for subject in personID:
        
        results_json_filename = os.path.join(folder_path, 'result' + f"_{subject:03d}.json")
        
        remaining_runs = check_completed_runs(results_json_filename, allrun)
        print(f"第{subject}位受试者的剩余计算任务：{len(remaining_runs)}个")
        
        # load data
        x, y, _ = paradigm.get_data(dataset, [subject])
        data = x[:, :, :-(x.shape[2] % fs)] if x.shape[2] % fs else x
        _, label = np.unique(y, return_inverse=True)
        
        all_data_list, _= split_data_for_cv(data, label)
              
        def process_run(irun):
            result = calc_acc(
                irun, 
                all_data_list[irun[0]]['traindata'], 
                all_data_list[irun[0]]['trainlabel'], 
                all_data_list[irun[0]]['testdata'], 
                all_data_list[irun[0]]['testlabel'],
                verbose=False,
                device_use = device_use
            )
            save_result(irun, result, results_json_filename)

        with parallel_backend('loky', n_jobs=n_jobs):
            Parallel(batch_size=1, verbose=len(remaining_runs))(
                delayed(process_run)(irun) for irun in remaining_runs
            )
    
    print(f"一共有 {len(all_tasks)} 个计算任务。")
    with parallel_backend('loky', n_jobs=n_jobs):
        Parallel(batch_size=1, verbose=len(all_tasks))( # type: ignore
            delayed(process_run)(task) for task in all_tasks
        )

#%% Program initialization settings
dataset_list = [Cho2017,Lee2019_MI,Pan2023,PhysionetMI,Shin2017A,Weibo2014]

# Defining the signal downsampling frequency
fs = 160

# Define the frequency range of the bandpass filter
freqband = [5, 32]

# Define the number of cross-validation folds and repetitions
nRepeat = 10
kfold = 10
seed = 42
kf = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=nRepeat, random_state=seed)

# Defining Filtering Methods
methods = ['none','rsf']

# Defining Filter Dimensions
min_dim = 2
max_dim = 22
step_dim = 2 

dims = list(range(min_dim,max_dim,step_dim))
allrun = list(itertools.product(range(nRepeat*kfold), methods, dims))

# Definition of classification methods
algorithms = ['EEGNetv4','ShallowFBCSPNet','Deep4Net','oFBCNet','oGraph_CSPNet','LMDANet']

# Define a folder for saving results
floder_name = 'Result_DL'
if not os.path.exists(floder_name):
    os.makedirs(floder_name)

# Setting the calculation mode
device_use = 'cuda' # 'cuda' or 'cpu'

# Setting the maximum number of parallel processes
cpu_jobs = 32
n_jobs = min(int(mp.cpu_count()), 1 if device_use == 'cuda' else cpu_jobs)
verbose = False if n_jobs > 1 else True

if __name__ == '__main__':
    
    # Defining GPU shared locks and values   
    manager = mp.Manager()
    gpu_lock = manager.Lock() 
    
    for dataset_ in dataset_list:
        paradigm = LeftRightImagery(resample=fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4) 
        dataset = dataset_()
        subjects = dataset.subject_list

        # Creating the results folder
        folder_path = os.path.join(floder_name, dataset)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Running the main process
        main_processes(dataset, subjects, allrun)

