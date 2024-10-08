# Riemannian Geometry-Based Spatial Filtering (RSF)

[中文版本](./README.ch.md)

## Introduction

**Riemannian Geometry-Based Spatial Filtering (RSF)** is a method based on Riemannian geometry designed to improve the accuracy of motor imagery (MI) and electroencephalogram (EEG) signal classification.

## Code Structure

- **Lib_develop**: Contains specially modified tool libraries used during our development process.
- **deep_learning**: Contains code related to deep learning models.
- **loaddata**: Contains code for loading and preprocessing data.
- **main.py**: The main execution file used to run the entire project.
- **main_dev.py**: The development version of the main execution file.
- **requirements.txt**: Lists all dependencies required to run the project.
- **rsf.py**: Contains the core code for implementing the RSF method.

## Installation Guide

To install and run the project, please follow these steps:

1. Clone the repository locally.
2. Install the necessary dependencies by running `pip install -r requirements.txt`.
3. Run `python main.py` to start the project.

## Related Research Resources

We express our gratitude to the open-source community, which facilitates the broader dissemination of research by other researchers and ourselves. The coding style in this repository is relatively rough. We welcome anyone to refactor it to make it more efficient. Our model codebase is largely based on the following repositories:

- [<img src="https://img.shields.io/badge/GitHub-MOABB-b31b1b"></img>](https://github.com/NeuroTechX/moabb) An open science project aimed at establishing a comprehensive benchmark for BCI algorithms using widely available EEG datasets.
- [<img src="https://img.shields.io/badge/GitHub-MetaBCI-b31b1b"></img>](https://github.com/TBC-TJU/MetaBCI) An open-source non-invasive brain-computer interface platform.
- [<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann) A Python library focused on Riemannian geometry methods for EEG signal classification. pyRiemann provides a suite of tools for processing and classifying EEG signals in Riemannian space.
- [<img src="https://img.shields.io/badge/GitHub-FBCNet-b31b1b"></img>](https://github.com/ravikiran-mane/FBCNet) A convolutional neural network based on filter banks for EEG signal classification. FBCNet combines traditional band-pass feature extraction with deep learning to improve classification performance.
- [<img src="https://img.shields.io/badge/GitHub-Braindecode-b31b1b"></img>](https://github.com/braindecode/braindecode) Contains several deep learning models such as EEGNet, ShallowConvNet, and DeepConvNet, designed specifically for EEG signal classification. Braindecode aims to provide an easy-to-use deep learning toolbox.
- [<img src="https://img.shields.io/badge/GitHub-CSPNet-b31b1b"></img>](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet) Contains Tensor-CSPNet and Graph-CSPNet, two deep learning models for MI-EEG signal classification.
- [<img src="https://img.shields.io/badge/GitHub-LMDANet-b31b1b"></img>](https://github.com/MiaoZhengQing/LMDA-Code) A deep learning-based network for EEG signal classification. LMDA-Net combines various advanced neural network architectures to enhance classification accuracy.

## Data Availability

We used the following public datasets:

- [<img src="https://img.shields.io/badge/DOI-Pan2023-blue"></img>](https://doi.org/10.7910/DVN/O5CQFA) Provides cross-session left/right hand MI-EEG data from 14 subjects. 
- [<img src="https://img.shields.io/badge/DOI-Cho2017-green"></img>](http://gigadb.org/dataset/100295) Provides left/right hand MI-EEG data from 52 subjects. 
- [<img src="https://img.shields.io/badge/DOI-Lee2019-orange"></img>](https://doi.org/10.1093/gigascience/giz002) Provides left/right hand MI-EEG data from 54 subjects. 
- [<img src="https://img.shields.io/badge/DOI-PhysioNet-red"></img>](https://www.physionet.org/content/eegmmidb/1.0.0/) Provides left/right hand MI-EEG data from 106/109 subjects. 
- [<img src="https://img.shields.io/badge/DOI-Shin2017-purple"></img>](http://doc.ml.tu-berlin.de/hBCI) Provides left/right hand MI-EEG data from 29 subjects. 
- [<img src="https://img.shields.io/badge/DOI-Yi2014-yellow"></img>](https://doi.org/10.7910/DVN/27306) Provides seven-class MI-EEG data from 10 subjects. 

**Table 1** Details of all public datasets

| Dataset                                                |     Classes     | Trials | Channels | Duration (s) | Subjects |
| :----------------------------------------------------- | :-------------: | :----: | :------: | :----------: | :------: |
| [Cho2017](https://doi.org/10.1093/gigascience/gix034)  | left/right hand |  200   |    64    |      3       |    52    |
| [Lee2019](https://doi.org/10.1093/gigascience/giz002)  | left/right hand |  200   |    62    |      4       |    54    |
| [Pan2023](https://doi.org/10.1088/1741-2552/ad0a01)    | left/right hand |  240   |    28    |      4       |    14    |
| [PhysioNet](https://doi.org/10.1109/TBME.2004.827072)  | left/right hand | 40-60  |    64    |      3       |   106    |
| [Shin2017](https://doi.org/10.1109/TNSRE.2016.2628057) | left/right hand |   60   |    30    |      4       |    29    |
| [Yi2014](https://doi.org/10.1371/journal.pone.0114853) | left/right hand |  160   |    60    |      4       |    10    |
| **Total:**                                             |                 |        |          |              | **265**  |


## License and Attribution

© 2024 Lc.Pan. All rights reserved.
Please refer to the [LICENSE](./LICENSE) file for details on the licensing of our code.
