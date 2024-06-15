import os
import numpy as np
from scipy.io import savemat
from sklearn.model_selection import RepeatedStratifiedKFold
from dataset_mi import Dataset_Left_Right_MI
from ..utils import create_folder


# 划分数据集
def split_data_for_cv(data, label):
    
    all_indices = {}
    for cv, (train_index, test_index) in enumerate(kf.split(data, label)):
        # 保存训练集和测试集的索引
        all_indices[cv] = {}
        all_indices[cv]['train_index'] = train_index
        all_indices[cv]['test_index'] = test_index 
    
    return all_indices


# dataset_list = ['BNCI2014_001','Cho2017','Weibo2014','Pan2023',
#                 'PhysionetMI','Lee2019_MI', 'Shin2017A']

dataset_list = ['PhysionetMI']
print(dataset_list)
fs = 160
period=[-4,4]
nRepeat = 10
kfold = 10
seed = 42
kf = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=nRepeat, random_state=seed)

for dataset_name in dataset_list:
    print(f'processing {dataset_name} dataset')
    dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=1,fmax=79,tmin=period[0],tmax=period[1])
    subjects = dataset.subject_list
    # subjects = np.arange(1,109)
    # subjects = np.delete(subjects, [87,91,99,103])
    # subjects = [101,102,103,105,106,107,108,109]

    folder_path = os.path.join('G:\dataset', dataset_name,'subjects')
    folder_path = os.path.join('G:\dataset', dataset_name,'cross_validated_indexes')
    create_folder(folder_path)
    
    badID = []
    for subject in subjects:
        try:
            data, label = dataset.get_data([subject])
        except:
            badID.append(subject)
            print(f'{subject}数据集存在问题！')
            continue
        
        data = data.transpose(1,2,0)
        label[label==1] = 2
        label[label==0] = 1
        
        file_name = os.path.join(folder_path, f's{subject:03d}.mat')
        file_name = os.path.join(folder_path, f's{subject:03d}_index_of_10x10_fold_cv.mat')
        
        savemat(file_name, 
                {'data':np.array(data,dtype=np.float64), 
                 'label':np.array(label,dtype=np.float64), 
                 'fs':np.float64(fs), 
                 'period':np.array(period,dtype=np.float64)},
                oned_as='column')
        
        all_indices= split_data_for_cv(data, label)
        
        savemat(file_name, 
                {'train_indices_list':[all_indices[cv]['train_index']+1 for cv in range(nRepeat*kfold)], 
                 'test_indices_list':[all_indices[cv]['test_index']+1 for cv in range(nRepeat*kfold)], 
                 'n_repeats':nRepeat,
                 'kfold':kfold,
                 'dataset':np.array(dataset_name,dtype=np.str_),
                 'subject':subject,
                 }, 
                oned_as='column')

        print(f'{subject} done')
        
    print(f'{dataset_name} dataset done')
    print(f'badID: {badID}')