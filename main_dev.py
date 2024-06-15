# This code is used for the paper "Enhancing Motor Imagery EEG Classification with a Riemannian Geometry-Based Spatial Filtering (RSF) Method"
# Authors: Pan.LC <panlincong@tju.edu.cn>
# Date: 2024/3/18
# License: MIT License

# 公共工具库
import os, time
import json
import numpy as np
import itertools
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import RepeatedStratifiedKFold
from contextlib import redirect_stdout, redirect_stderr

# 私有工具库
from loaddata import Dataset_Left_Right_MI
from deep_learning.dl_classifier import DL_Classifier
from rsf import RSF

# 计算单个任务的函数
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
    
    # 训练模型并测试
    for model_name in algorithms:
        
        # 训练模型并测试
        if not verbose:
            # 这行代码用于隐藏程序执行过程中的输出显示
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                testacc, elapsed_time = train_and_evaluate(traindata, trainlabel, testdata, testlabel, 
                                                           model_name, device, irun=irun)
        else:
            testacc, elapsed_time = train_and_evaluate(traindata, trainlabel, testdata, testlabel, 
                                                       model_name, device, irun=irun)
        
        # 记录结果
        Acc.append(testacc)
        Timecost.append(elapsed_time + transform_time)
              
    return Acc, Timecost

# 训练模型并测试的函数
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
        #将错误信息保存到文件
        with open(os.path.join(floder_name, 'error_{}.txt'.format(dataset_name)), 'w') as f:
            # 加上时间戳
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")
            # 将错误信息写入文件
            f.write(f"Error in {model_name} with {rsf_method} and {rsf_dim} on run {irun[0]}: {e}\n")
    return testacc, elapsed_time

# 保存单个计算结果的函数
def save_result(irun, result, filename):
    # 将结果转换为字典并保存为JSON格式
    result_dict = {'run': irun[0], 'method': irun[1], 'dim': irun[2], 'Acc': result[0], 'Timecost': result[1]}
    try:
        with gpu_lock:
            with open(filename, 'a') as f:
                json.dump(result_dict, f)
                f.write('\n')  # 换行，以便于读取时分割
    except:
        with open(filename, 'a') as f:
                json.dump(result_dict, f)
                f.write('\n')  # 换行，以便于读取时分割

# 检查已完成的计算并返回未完成的计算列表
def check_completed_runs(filename, allrun):
    if not os.path.exists(filename):
        return allrun
    completed_runs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 检查文件是否包含子标题部分
    results_start = lines.index("results:\n") + 1 if "results:\n" in lines else 0
    for line in lines[results_start:]:
        if line.strip():
            completed_runs.append(json.loads(line))
    completed_indices = [(run['run'], run['method'], run['dim']) for run in completed_runs]
    return [irun for irun in allrun if (irun[0], irun[1], irun[2]) not in completed_indices]
         
# 划分数据集
def split_data_for_cv(data, label):
    all_data = {}
    all_indices = {}
    for cv, (train_index, test_index) in enumerate(kf.split(data, label)):
        # 保存训练集和测试集的索引
        all_indices[cv] = {}
        all_indices[cv]['train_index'] = train_index
        all_indices[cv]['test_index'] = test_index
        # 保存训练集和测试集的数据
        all_data[cv] = {}
        all_data[cv]['traindata'] = data[train_index]
        all_data[cv]['trainlabel'] = label[train_index]
        all_data[cv]['testdata'] = data[test_index]
        all_data[cv]['testlabel'] = label[test_index]  
    return all_data, all_indices

# 主函数
def main_processes(dataset, personID, allrun):
    
    all_tasks = []
    for subject in personID:
        
        # 设置中间计算结果JSON文件名
        results_json_filename = os.path.join(folder_path, 'result' + f"_{subject:03d}.json")
        
        # 检查剩余的计算任务
        remaining_runs = check_completed_runs(results_json_filename, allrun)
        print(f"第{subject}位受试者的剩余计算任务：{len(remaining_runs)}个")
        
        # 读取数据集
        data, label = dataset.get_data([subject])
        all_data_list, _= split_data_for_cv(data, label)
              
        # 执行剩余的计算   
        # 使用 parallel_backend 设置并行计算的后端
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
    
    # 执行剩余的计算   
    # 使用 parallel_backend 设置并行计算的后端，'loky'为多进程，'multiprocessing'为多线程
    print(f"一共有 {len(all_tasks)} 个计算任务。")
    with parallel_backend('loky', n_jobs=n_jobs):
        Parallel(batch_size=1, verbose=len(all_tasks))( # type: ignore
            delayed(process_run)(task) for task in all_tasks
        )

#%% 程序初始化设置
# dataset_list = ['Cho2017','Lee2019_MI','Pan2023','PhysionetMI','Shin2017A','Weibo2014']
dataset_list = ['Cho2017']

# 定义信号降采样频率
fs = 160

# 定义带通滤波器的频率范围(如果执行)
freqband = [5, 32]

# 定义交叉验证折数和重复次数
nRepeat = 10
kfold = 10
seed = 42
kf = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=nRepeat, random_state=seed)

# 定义滤波器的计算方法
methods = ['none','rsf']

# 定义滤波器维度和间隔
min_dim = 2
max_dim = 22
step_dim = 2 

# 定义分类方法
algorithms = ['EEGNetv4','ShallowFBCSPNet','Deep4Net','FBCNet','Graph_CSPNet','LMDANet']

# 定义保存结果文件夹
floder_name = 'Result_DL'
if not os.path.exists(floder_name):
    os.makedirs(floder_name)

# 设置计算模式
device_use = 'cuda' # 'cuda' or 'cpu'

# 设置最大并行进程数
cpu_jobs = 32
n_jobs = min(int(mp.cpu_count()), 1 if device_use == 'cuda' else cpu_jobs)
verbose = False if n_jobs > 1 else True

#%% 运行主函数
if __name__ == '__main__':
    
    # 定义GPU共享锁和值   
    manager = mp.Manager()
    gpu_lock = manager.Lock() 
    
    for dataset_name in dataset_list:
        # 设置数据集和参数
        dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4)
        subjects = dataset.subject_list
        n_chan = dataset.n_channels
        dims = list(range(min_dim,np.min([n_chan, max_dim]),step_dim))
        allrun = list(itertools.product(range(nRepeat*kfold), methods, dims))
        
        # 创建结果文件夹
        folder_path = os.path.join(floder_name, dataset_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 运行主函数
        personID = subjects
        main_processes(dataset, personID, allrun)

