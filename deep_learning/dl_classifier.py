"""
DeepL_Classifier: Deep Learning Classifier for EEG Data
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/3/15
License: BSD 3-Clause License

Introduction:
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
        'ShallowNet', 'ShallowFBCSPNet', 'DeepNet', 'Deep4Net' 
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
    returns:
        self: The fitted model.

predict(X): Predicts labels for the given data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).
    returns:
        predictions (array-like): Predicted labels with shape (n_samples,).

score(X, y): Computes the accuracy of the model on the given data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).
    y (array-like): True labels for computing accuracy.
    returns:
        accuracy (float): Accuracy of the model on the given data.

Input/Output Details:
    1> Input data (X) should be a 3D NumPy array or any array-like structure compatible with PyTorch, 
       with dimensions corresponding to (samples, channels, time points).
    2> Output predictions are NumPy arrays containing the predicted labels for each sample.
    3> If true labels (y) are provided during the prediction process, the method also returns the accuracy as a float.
    
Please ensure that the input data is preprocessed and compatible with the model requirements. 
The sampling frequency (fs) should match the frequency used during data collection.

This documentation provides an overview of the DL_Classifier class, its methods, and how to use it for EEG data 
classification tasks. For more detailed information on the individual models and their specific requirements, 
refer to the respective model documentation.

Example:
from dl_classifier import DL_Classifier

# Initialize the classifier with EEGNet model and training parameters
classifier = DeepL_Classifier(model_name='EEGNet', fs=128, batch_size=32, lr=1e-2, max_epochs=200, device='cuda')

# Fit the classifier to the training data
classifier.fit(train_data, train_labels)

# Predict labels for the test data
test_predictions = classifier.predict(test_data)

# Compute accuracy on the test data
test_accuracy = classifier.score(test_data, test_labels)

"""

import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from . import (EEGNet, ShallowNet, DeepNet,         
               EEGNetv4, ShallowFBCSPNet, Deep4Net,
               FBCNet, Tensor_CSPNet, Graph_CSPNet, 
               LMDANet)
from .format_data import Formatdata


def check_nn(est):
    """Check if a given estimator is valid."""
    estimators = {
        'EEGNet': EEGNet,
        'EEGNetv4': EEGNetv4,
        'ShallowNet': ShallowNet,
        'ShallowFBCSPNet': ShallowFBCSPNet,
        'DeepNet': DeepNet,
        'Deep4Net': Deep4Net,  
        'FBCNet': FBCNet,
        'oFBCNet': FBCNet,
        'Tensor_CSPNet': Tensor_CSPNet,
        'oTensor_CSPNet': Tensor_CSPNet,
        'Graph_CSPNet': Graph_CSPNet,
        'oGraph_CSPNet': Graph_CSPNet,
        'LMDANet': LMDANet,
    }

    if callable(est):
        pass
    elif est in estimators.keys():
        est = estimators[est]
    else:
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
            callable function""" % (est, (' , ').join(estimators.keys())))
    return est


class DL_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='EEGNet', n_classes=2, fs=128, batch_size=32, lr=1e-2, max_epochs=200, device='cpu', **kwargs):
        self.model_name = model_name
        self.n_classes = n_classes
        self.fs = fs
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = 'cuda' if device == 'gpu' else device
        self.pp_name = None
        self.nn_name = None
        self.Process = None
        self.Net = None
        self.Model = None
        self.rsf_method = kwargs.get('rsf_method', 'none') 
        self.rsf_dim = kwargs.get('rsf_dim', 4)
        self.freqband = kwargs.get('freqband', None)
        self.dtype = kwargs.get('dtype', 'float32')
        self.seed = kwargs.get('seed', 42)
        self.patience = kwargs.get('patience', 50)
        self.net_params = {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'max_epochs': self.max_epochs,
            'device': self.device,
            'dtype': self.dtype,
            'seed': self.seed,
            'patience': self.patience,
            # **kwargs  # This allows for any additional parameters to be passed
        }

    def fit(self, X, y):
        X = X.copy()
        self.pp_name = self.model_name.split('-')[0] if '-' in self.model_name else 'None'
        self.nn_name = self.model_name.split('-')[-1] if '-' in self.model_name else self.model_name
        
        Process = None
        if self.pp_name.lower() == 'rsf' or self.nn_name in ['Graph_CSPNet', 'Tensor_CSPNet', 'FBCNet']:
            Process = Formatdata(fs=self.fs, n_times=X.shape[2], alg_name=self.nn_name, dtype=self.dtype,   
                                 rsf_method=self.rsf_method, rsf_dim=self.rsf_dim, freqband=self.freqband)
            X = Process.fit_transform(X, y)
        
        Network = check_nn(self.nn_name)
        if self.nn_name in ['Graph_CSPNet']:
            graph_M = Process.graph_M.to(self.device)
            Net = Network(graph_M, X.shape[1], X.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        elif self.nn_name in ['Tensor_CSPNet']:
            Net = Network(len(Process.time_seg), X.shape[1] * X.shape[2], X.shape[3], n_classes = self.n_classes, 
                          net_params=self.net_params)
        elif self.nn_name in ['FBCNet']:
            Net = Network(X.shape[2], X.shape[3], n_classes = self.n_classes, net_params=self.net_params)
        else:
            Net = Network(X.shape[1], X.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        

        Net.fit(X.astype(self.dtype).copy(), y.astype('int64').copy())
        
        self.Process = Process
        self.Net = Net
        self.Model = make_pipeline(Process, Net) if Process is not None else Net
        
        return self

    def predict(self, X):
        if hasattr(self, 'Model'):
            predictions = self.Model.predict(X.astype(self.dtype).copy())
        else:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'predict'.")
        return predictions

    def score(self, X, y):
        if hasattr(self, 'Model'):
            y_pred = self.predict(X)
            return accuracy_score(y.astype('int64'), y_pred)
        else:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'score'.")
    
