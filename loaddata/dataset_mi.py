# This file is used to load the dataset and preprocess the data. 
# Authors: Pan.LC <panlincong@tju.edu.cn>
# Date: 2024/4/7

# Pan et al. 对moabb库进行了一些可有可无的调整,用于灵活的修改数据集的保存地址,但这是不必要的。
# 可以使用原版moabb库和pan2023.py文件中的函数来加载Pan2023数据集。
from moabb.paradigms import LeftRightImagery, MotorImagery
from moabb.datasets import (BNCI2014_001,
                            BNCI2014_002,
                            BNCI2014_004,
                            BNCI2015_001,
                            BNCI2015_004,
                            Schirrmeister2017,
                            Cho2017,
                            AlexMI,
                            Weibo2014,
                            PhysionetMI,
                            Zhou2016,
                            GrosseWentrup2009,
                            Lee2019_MI,
                            Shin2017A,)

from .pan2023 import Pan2023
import numpy as np

def dataset_loader(dataset_name, subjects, new_fs = 160):
    # 定义数据集和对应的采样频率
    datasets = {
        "BNCI2014_001": (BNCI2014_001, 250),
        "Cho2017": (Cho2017, 512),
        "GrosseWentrup2009": (GrosseWentrup2009, 500),
        "Lee2019_MI": (Lee2019_MI, 1000),
        "PhysionetMI": (PhysionetMI, 160),
        "Schirrmeister2017": (Schirrmeister2017, 500),
        "Shin2017A": (Shin2017A, 200),
        "Weibo2014": (Weibo2014, 200),
        "Zhou2016": (Zhou2016, 250),
        'Pan2023': (Pan2023, 250),
    }

    # 检查数据集名称是否有效
    if dataset_name not in datasets:
        raise ValueError("Invalid dataset name")

    # 实例化数据集并获取采样频率
    dataset_class, fs = datasets[dataset_name]
    dataset = dataset_class() if dataset_name != 'Shin2017A' else dataset_class(accept=True)
    
    # 检查subjects是否完全属于subject_list
    subject_list = dataset.subject_list
    if not set(subjects).issubset(set(subject_list)):
        print(f"dataset: {dataset_name}, valid subjects: {subject_list}, entered subjects: {subjects}")
        raise ValueError("Invalid subject numbers were entered!")

    # 获取数据
    paradigm = LeftRightImagery(resample=new_fs if new_fs is not None else fs) 
    
    x, y, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
    
    # 有时会出现数据长度超出预期1个时间点，因此需要减去多余的点
    data = x[:, :, :-(x.shape[2] % new_fs)] if x.shape[2] % new_fs else x
    
    unique_label, label = np.unique(y, return_inverse=True)
    
    return data, label, new_fs


class Dataset_MI:
    """Single Bandpass filter motor Imagery.

    Motor imagery paradigm with only one bandpass filter (default 8 to 32 Hz)

    Parameters
    ----------
    fmin: float (default 8)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 32)
        cutoff frequency (Hz) for the low pass filter
    
    * if fmin and fmax are not provided, default to 8 to 32 Hz.
    * if fmin is greater than fmax, swap them.

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline
            correction. If None, do not apply baseline correction.
            If a tuple (a, b), the interval is between a and b (in seconds),
            including the endpoints.
            Correction is applied by computing the mean of the baseline period
            and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    
    ** Pan.LC edited on 2024/4/7 额外添加的参数和功能，仅适用于Pan.LC改动后的moabb环境（原版moabb环境无法使用）
    path: str | None (default None)
        path to the dataset folder. If None, use the default path.
    
    if fmin == fmax, no filter is applied.
    
    """
    def __init__(self, dataset_name: str, fs=None, **kwargs):
        # 定义数据集和对应的采样频率、通道数
        datasets = {
            "BNCI2014_001": (BNCI2014_001, 250, 22, 4),
            "BNCI2014_002": (BNCI2014_002, 512, 15, 5),
            "BNCI2014_004": (BNCI2014_004, 250, 3, 4.5),
            "BNCI2015_001": (BNCI2015_001, 512, 13, 5),
            "BNCI2015_004": (BNCI2015_004, 256, 30, 7),
            "AlexMI": (AlexMI, 512, 16, 3),
            "Cho2017": (Cho2017, 512, 64, 3),
            "GrosseWentrup2009": (GrosseWentrup2009, 500, 128, 7),
            "Lee2019_MI": (Lee2019_MI, 1000, 62, 4),
            "PhysionetMI": (PhysionetMI, 160, 64, 3),
            "Schirrmeister2017": (Schirrmeister2017, 500, 128, 4),
            "Shin2017A": (Shin2017A, 200, 30, 10),
            "Weibo2014": (Weibo2014, 200, 60, 4),
            "Zhou2016": (Zhou2016, 250, 14, 5),
            'Pan2023': (Pan2023, 250, 28, 4),
        }

        # 检查数据集名称是否有效
        if dataset_name not in datasets:
            raise ValueError("Invalid dataset name")
        
        # 实例化数据集并获取采样频率
        path = kwargs.pop('path', None)
        dataset_class, srate, ch_num, time_length = datasets[dataset_name]
        self.dataset = dataset_class(path=path) if dataset_name != 'Shin2017A' else dataset_class(accept=True, path=path)
        self.fs = fs if fs is not None else srate
        tmax = kwargs.pop('tmax', time_length)
        kwargs['tmax'] = np.min([tmax, time_length])
        self.paradigm = MotorImagery(resample=self.fs, **kwargs) # 降采样频率为160Hz
        self.subject_list = self.dataset.subject_list
        self.n_channels = ch_num
        self.time_length = time_length
        
        if dataset_name == 'PhysionetMI':
            # 由于PhysionetMI数据集的88, 92, 100等三个编号的subject的数据存在问题，因此需要将其剔除
            self.subject_list = [x for x in self.subject_list if x not in [88, 92, 100]]
    
    def get_data(self, subjects: list[int]):
        # 检查subjects是否完全属于subject_list
        if not set(subjects).issubset(set(self.subject_list)):
            print(f"dataset: {self.dataset.code}, valid subjects: {self.subject_list}, entered subjects: {subjects}")
            raise ValueError("Invalid subject numbers were entered!")

        # 获取数据
        x, y, _ = self.paradigm.get_data(dataset=self.dataset, subjects=subjects)
        
        # 由于浮点数问题，可能会出现数据长度超出预期1个时间点，因此需要减去多余的点
        data = x[:, :, :-(x.shape[2] % self.fs)] if x.shape[2] % self.fs else x
        
        self.events, label = np.unique(y, return_inverse=True)
        
        return data, label

class Dataset_Left_Right_MI(Dataset_MI):
    def __init__(self, dataset_name, fs=None, **kwargs):
        # 定义数据集和对应的采样频率
        datasets = {
            "BNCI2014_001": (BNCI2014_001, 250, 22, 4),            #  9 subjects
            "Cho2017": (Cho2017, 512, 64, 3),                      # 52 subjects
            "GrosseWentrup2009": (GrosseWentrup2009, 500, 128, 7), # 10 subjects
            "Lee2019_MI": (Lee2019_MI, 1000, 62, 4),               # 54 subjects
            "PhysionetMI": (PhysionetMI, 160, 64, 3),              #109 subjects
            "Schirrmeister2017": (Schirrmeister2017, 500, 128, 4), # 14 subjects
            "Shin2017A": (Shin2017A, 200, 30, 10),                 # 29 subjects
            "Weibo2014": (Weibo2014, 200, 60, 4),                  # 10 subjects
            "Zhou2016": (Zhou2016, 250, 14, 5),                    #  4 subjects
            'Pan2023': (Pan2023, 250, 28, 4),                      # 14 subjects
        }
        # 检查数据集名称是否有效
        if dataset_name not in datasets:
            raise ValueError("Invalid dataset name")
        
        # 实例化数据集并获取采样频率
        path = kwargs.pop('path', None) 
        dataset_class, srate, ch_num, time_length = datasets[dataset_name]
        self.dataset = dataset_class(path=path) if dataset_name != 'Shin2017A' else dataset_class(accept=True, path=path)
        self.fs = fs if fs is not None else srate
        tmax = kwargs.pop('tmax', time_length)
        kwargs['tmax'] = np.min([tmax, time_length])
        self.paradigm = LeftRightImagery(resample=self.fs, **kwargs) # 降采样频率为160Hz    
        self.subject_list = self.dataset.subject_list
        self.n_channels = ch_num
        self.time_length = time_length
        
        if dataset_name == 'PhysionetMI':
            # 由于PhysionetMI数据集的88, 92, 100等三个编号的subject的数据存在问题，因此需要将其剔除
            self.subject_list = [x for x in self.subject_list if x not in [88, 92, 100]]
        
        
