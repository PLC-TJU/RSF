"""
DeepL_Classifier: Deep Learning Classifier for EEG Data
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/3/14
License: BSD 3-Clause License

The DeepL_Classifier is a Python class designed to facilitate the training and evaluation of various deep 
learning models for electroencephalogram (EEG) data classification. It is built on top of PyTorch and 
scikit-learn, providing a flexible and easy-to-use interface for experimenting with different neural 
network architectures.

Features:
    1> Supports multiple EEG-based deep learning models.
    2> Integrates with scikit-learn's BaseEstimator and TransformerMixin for compatibility with scikit-learn workflows.
    3> Allows customization of training parameters such as batch size, learning rate, and number of epochs.
    4> Can be used with any device that PyTorch supports (CPU or CUDA-enabled GPU).

Usage:
To use the DeepL_Classifier, you need to initialize it with the desired model name and training parameters. 
Then, you can fit the model to your training data and use it to transform (predict) on new data.

Initialization Parameters:
    1> model_name (str): Name of the deep learning model to use. Supported models include: 
        'ShallowNet', 'ShallowFBCSPNet'
        'DeepNet'
                R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding and 
                visualization," Hum Brain Mapp, vol. 38, no. 11, pp. 5391-5420, Nov 2017, doi: 10.1002/hbm.23730.
        'EEGNet','EEGNetv4'      
                V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based brain-computer
                interfaces," J Neural Eng, vol. 15, no. 5, p. 056013, Oct 2018, doi: 10.1088/1741-2552/aace8c.
        'FBCNet'
                R. Mane et al., "FBCNet: A multi-view convolutional neural network for brain-computer interface," 
                arXiv preprint, vol. 2104.01233, Mar 2021. [Online]. Available: https://arxiv.org/abs/2104.01233.
        'Tensor_CSPNet'
                C. Ju and C. Guan, "Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery 
                Classification," IEEE Trans Neural Netw Learn Syst, vol. 34, no. 12, pp. 10955-10969, Dec 2023, 
                doi: 10.1109/TNNLS.2022.3172108.
        'Graph_CSPNet'
                C. Ju and C. Guan, "Graph Neural Networks on SPD Manifolds for Motor Imagery Classification: A 
                Perspective From the Time-Frequency Analysis," IEEE Trans Neural Netw Learn Syst, vol. PP, pp. 1-15, 
                Sep 19 2023, doi: 10.1109/TNNLS.2023.3307470.
        'LMDANet'
                Z. Miao, M. Zhao, X. Zhang, and D. Ming, "LMDA-Net:A lightweight multi-dimensional attention network
                for general EEG-based brain-computer interfaces and interpretability," Neuroimage, vol. 276, p. 120209,
                Aug 1 2023, doi: 10.1016/j.neuroimage.2023.120209.
             
    2> fs (int): Sampling frequency of the EEG data.
    3> batch_size (int): Number of samples per batch during training.
    4> lr (float): Learning rate for the optimizer.
    5> max_epochs (int): Maximum number of epochs for training.
    6> device (str): Device to run the computations on ('cpu' or 'cuda').
    7> **kwargs: Additional keyword arguments to pass to the underlying neural network.

Methods:
fit(X, y): Trains the model on the provided data.
    X (array-like): Training data with shape (n_samples, n_channels, n_times).
    y (array-like): Target labels with shape (n_samples,).

transform(X): Transforms on the provided data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).

predict(X): Predicts labels for the given data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).

score(X, y): Computes the accuracy of the model on the given data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).
    y (array-like): True labels for computing accuracy.

Input/Output Details:
    1> Input data (X) should be a 3D NumPy array or any array-like structure compatible with PyTorch, 
       with dimensions corresponding to (samples, EEG channels, time points).
    2> Output predictions are NumPy arrays containing the predicted labels for each sample.
    3> If true labels (y) are provided during the transform, the method also returns the accuracy as a float.
    
Please ensure that the input data is preprocessed and compatible with the model requirements. 
The sampling frequency (fs) should match the frequency used during data collection.

This documentation provides an overview of the DL_Classifier class, its methods, and how to use it for EEG data 
classification tasks. For more detailed information on the individual models and their specific requirements, 
refer to the respective model documentation.

Example:
from dl_classifier import DL_Classifier

# Initialize the classifier with EEGNet model and training parameters
classifier = DeepL_Classifier(model_name='EEGNet', fs=128, batch_size=32, lr=1e-2, max_epochs=200, device='cpu')

# Fit the classifier to the training data
classifier.fit(train_data, train_labels)

# Transform (predict) on the test data
predictions, accuracy = classifier.transform(test_data, test_labels)

"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from . import (EEGNet, ShallowNet, DeepNet,
               EEGNetv4, ShallowFBCSPNet, Deep4Net,  #推荐使用
               FBCNet, Tensor_CSPNet, Graph_CSPNet, 
               LMDANet, Formatdata)

class DL_Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='EEGNet', n_classes=2, fs=128, batch_size=64, lr=1e-2, max_epochs=200, device='cpu', **kwargs):
        self.model_name = model_name
        self.n_classes = n_classes
        self.fs = fs
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = 'cuda' if device == 'gpu' else device
        self.model = None
        self.Fd_transformer = None
        self.net_params = {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'max_epochs': self.max_epochs,
            'device': self.device,
            # **kwargs  # This allows for any additional parameters to be passed
        }
        self.rsf_method = kwargs.get('rsf_method', 'none')  # Provide a default value if key is not present
        self.rsf_dim = kwargs.get('rsf_dim', 4)
        self.freqband = kwargs.get('freqband', [5,32])

    def fit(self, X, y):
        # Selection of different calculations and models based on model name
        if self.model_name == 'Tensor_CSPNet':
            self.Fd_transformer = Formatdata(fs=self.fs, n_times=X.shape[2], alg_name='Tensor_CSPNet', 
                                             rsf_method=self.rsf_method, rsf_dim=self.rsf_dim)    
            X_transformed = self.Fd_transformer.fit_transform(X, y)
            self.model = Tensor_CSPNet(len(self.Fd_transformer.time_seg), X_transformed.shape[1] * X_transformed.shape[2], 
                                       X_transformed.shape[3], n_classes = self.n_classes, net_params=self.net_params)
        
        elif self.model_name == 'Graph_CSPNet':
            self.Fd_transformer = Formatdata(fs=self.fs, n_times=X.shape[2], alg_name='Graph_CSPNet', 
                                             rsf_method=self.rsf_method, rsf_dim=self.rsf_dim)
            X_transformed = self.Fd_transformer.fit_transform(X, y)
            graph_M = self.Fd_transformer.graph_M.to(self.device)
            self.model = Graph_CSPNet(graph_M, X_transformed.shape[1], X_transformed.shape[2], n_classes = self.n_classes, 
                                      net_params=self.net_params)
        
        elif self.model_name == 'FBCNet':
            self.Fd_transformer = Formatdata(fs=self.fs, n_times=X.shape[2], alg_name='FBCNet',
                                             rsf_method=self.rsf_method, rsf_dim=self.rsf_dim)
            X_transformed = self.Fd_transformer.fit_transform(X, y)#trial x 1 x chan x time x filterBand
            self.model = FBCNet(X_transformed.shape[2], X_transformed.shape[3], n_classes = self.n_classes, net_params=self.net_params)
        
        elif self.model_name == 'RSF-LMDANet':
            self.Fd_transformer = Formatdata(fs=self.fs, n_times=X.shape[2], alg_name='LMDANet',
                                             rsf_method=self.rsf_method, rsf_dim=self.rsf_dim, freqband=self.freqband)
            X_transformed = self.Fd_transformer.fit_transform(X, y)
            self.model = LMDANet(X_transformed.shape[1], X_transformed.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        
        elif self.model_name == 'LMDANet':
            X_transformed = X
            self.model = LMDANet(X_transformed.shape[1], X_transformed.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        
        elif self.model_name == 'EEGNet':
            X_transformed = X
            self.model = EEGNet(X_transformed.shape[1], X_transformed.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        
        elif self.model_name == 'ShallowNet':
            X_transformed = X
            self.model = ShallowNet(X_transformed.shape[1], X_transformed.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        
        elif self.model_name == 'DeepNet':
            X_transformed = X
            self.model = DeepNet(X_transformed.shape[1], X_transformed.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        elif self.model_name == 'EEGNetv4':
            X_transformed = X
            self.model = EEGNetv4(X_transformed.shape[1], X_transformed.shape[2], n_outputs = self.n_classes, net_params=self.net_params)
        elif self.model_name == 'ShallowFBCSPNet':
            X_transformed = X
            self.model = ShallowFBCSPNet(X_transformed.shape[1], X_transformed.shape[2], n_outputs = self.n_classes, net_params=self.net_params)
        elif self.model_name == 'Deep4Net':
            X_transformed = X
            self.model = Deep4Net(X_transformed.shape[1], X_transformed.shape[2], n_outputs = self.n_classes, net_params=self.net_params)
        
        # training model
        # self.model.to(self.device)
        # self.model.fit(torch.from_numpy(X_transformed).to(self.device), torch.from_numpy(y).to(self.device))
        self.model.fit(X_transformed, y)

        return self

    def transform(self, X):
        # Converted data
        if self.Fd_transformer:
            X_transformed = self.Fd_transformer.transform(X)
        else:
            X_transformed = X

        return X_transformed

    def predict(self, X):
        # 确保模型已经训练
        if self.model is None:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'predict'.")
        # 转换数据
        X_transformed = self.transform(X)
        # 使用模型进行预测
        predictlabel = self.model.predict(X_transformed)
        # 返回预测结果
        return predictlabel
    
    def score(self, X, y):
        # 使用模型进行预测
        y_pred = self.predict(X)
        # 计算并返回准确率
        return accuracy_score(y, y_pred)