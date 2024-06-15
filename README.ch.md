# Riemannian Geometry-Based Spatial Filtering (RSF)

[English Version](./README.md)

## 简介

**Riemannian Geometry-Based Spatial Filtering (RSF)** 是一种基于黎曼几何的空间滤波方法，旨在提高电机想象（MI）和脑电图（EEG）信号分类的准确性。

## 代码结构

- **Lib_develop**: 包含我们开发过程中使用过的经过特殊修改的工具库。
- **deep_learning**: 包含深度学习模型相关的代码。
- **loaddata**: 包含加载和预处理数据的代码。
- **main.py**: 主执行文件，用于运行整个项目。
- **main_dev.py**: 开发版本的主执行文件。
- **requirements.txt**: 列出项目运行所需的所有依赖项。
- **rsf.py**: 包含实现RSF方法的核心代码。

## 安装指南

要安装和运行该项目，请按照以下步骤操作：

1. 克隆仓库到本地。
2. 通过运行 `pip install -r requirements.txt` 安装必要的依赖项。
3. 运行 `python main.py` 启动项目。

## 相关研究资源

我们对开源社区表示感谢，它为更广泛地传播研究成果提供了便利。本仓库中的代码风格相对粗糙，欢迎任何人对其进行重构以提高效率。我们的模型代码库在很大程度上基于以下资源库：

- [<img src="https://img.shields.io/badge/GitHub-MOABB-b31b1b"></img>](https://github.com/NeuroTechX/moabb) 这是一个开放科学项目，旨在建立一个包含广泛可用的EEG数据集的BCI算法的全面基准测试。
- [<img src="https://img.shields.io/badge/GitHub-MetaBCI-b31b1b"></img>](https://github.com/TBC-TJU/MetaBCI) 一个开源的非侵入式脑计算机接口平台。
- [<img src="https://img.shields.io/badge/GitHub-pyRiemann-b31b1b"></img>](https://github.com/pyRiemann/pyRiemann) 一个专注于黎曼几何方法的Python库，用于EEG信号分类。pyRiemann提供了一系列工具，用于处理和分类黎曼空间中的EEG信号。
- [<img src="https://img.shields.io/badge/GitHub-FBCNet-b31b1b"></img>](https://github.com/ravikiran-mane/FBCNet) 一个基于滤波器组的卷积神经网络，用于EEG信号分类。FBCNet结合了传统的频带特征提取和深度学习，以提高分类性能。
- [<img src="https://img.shields.io/badge/GitHub-Braindecode-b31b1b"></img>](https://github.com/braindecode/braindecode) 包含EEGNet、ShallowConvNet和DeepConvNet等多个深度学习模型，这些模型专为EEG信号分类设计。Braindecode旨在提供一个易于使用的深度学习工具箱。
- [<img src="https://img.shields.io/badge/GitHub-CSPNet-b31b1b"></img>](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet) 包含Tensor-CSPNet和Graph-CSPNet两个深度学习模型，用于MI-EEG信号分类。
- [<img src="https://img.shields.io/badge/GitHub-LMDANet-b31b1b"></img>](https://github.com/MiaoZhengQing/LMDA-Code) 一个基于深度学习的EEG信号分类网络。

## 数据可用性

我们使用了以下公开数据集：

- [<img src="https://img.shields.io/badge/DOI-Pan2023-blue"></img>](https://doi.org/10.7910/DVN/O5CQFA) 提供了14名受试者跨会话的左/右手MI-EEG数据。
- [<img src="https://img.shields.io/badge/DOI-Cho2017-green"></img>](http://gigadb.org/dataset/100295) 提供了52名受试者的左/右手MI-EEG数据。
- [<img src="https://img.shields.io/badge/DOI-Lee2019-orange"></img>](https://doi.org/10.1093/gigascience/giz002) 提供了54名受试者的左/右手MI-EEG数据。
- [<img src="https://img.shields.io/badge/DOI-Physionet-red"></img>](https://www.physionet.org/content/eegmmidb/1.0.0/) 提供了106/109名受试者的左/右手MI-EEG数据。
- [<img src="https://img.shields.io/badge/DOI-Shin2017-purple"></img>](http://doc.ml.tu-berlin.de/hBCI) 提供了29名受试者的左/右手MI-EEG数据。
- [<img src="https://img.shields.io/badge/DOI-Yi2014-yellow"></img>](https://doi.org/10.7910/DVN/27306) 提供了10名受试者的七种类别的MI-EEG数据。

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

## 许可和署名

版权 © 2024 年。保留所有权利。
请参阅 [LICENSE](./LICENSE) 文件，了解我们代码的许可情况。
