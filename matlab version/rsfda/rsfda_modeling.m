%% RSFDA Stacking集成学习模型训练程序
% Author: LC Pan
% Date: May. 4, 2025
% [1] Lc Pan, et al. Cross-session motor imagery-electroencephalography
% decoding with Riemannian spatial filtering and domain adaptation[J].
% Journal of Biomedical Engineering, 2025, 42(2):272-279.

function model = rsfda_modeling(Xs, Ys, Xt, Yt, fs, times, freqs, chans, varargin)
% STACKING_TRAIN 训练一个基于RSFDA的多时间窗、频带和通道的Stacking集成模型
% 使用并行计算加速基模型训练

% 输入:
%   Xs,Xt: 源域、目标域EEG数据 (通道×时间点×样本数)
%   Ys,Yt: 源域、目标域样本标签 (样本数×1)
%   fs: 采样率 (Hz)
%   times: 时间窗列表 (M×2数组, 单位:秒)
%   freqs: 频带列表 (N×2数组, 单位:Hz)
%   chans: 通道元组 (元胞数组, 每个元素是通道索引列表)
%   varargin: 可选参数
% 输出:
%   model: 训练好的Stacking模型

% 默认设置
if ~exist('fs','var') || isempty(fs)
    fs =250;
end
if ~exist('freqs','var') || isempty(freqs)
    freqs=[8,13;13,18;18,26;23,30;8,30];
end
if ~exist('times','var') || isempty(times) || isscalar(times)
    if isscalar(times)
        maxtime=times;
    else
        maxtime=size(Xs,2)/fs;
    end
    if maxtime>=4
        times=[0,2;1,3;2,4;0,3;1,4;0,4];
    elseif maxtime>=3
        times=[0,2;1,3;0,3];
    elseif maxtime>=2
        times=[0,1.5;0.5,2;0,2];
    else
        times=[0,maxtime];
    end
end
if ~exist('chans','var') || isempty(chans)
    chans={1:size(Xs,1)};
end
if ~iscell(chans)
    chans={chans};
    warning('chans参数必须是cell格式');
end

% 解析可选参数
p = inputParser;
addParameter(p, 'ClassifierType', 'LOGISTIC', @ischar);
addParameter(p, 'UseDecisionValues', true, @islogical);
addParameter(p, 'Optimize', false, @islogical);
addParameter(p, 'OptimizeTimeLimit', 30, @isnumeric);
addParameter(p, 'Verbose', false, @islogical);
addParameter(p, 'UseParallel', true, @islogical); % 添加并行计算选项
parse(p, varargin{:});

% 初始化模型结构
model = struct();
model.name = 'RSFDA';
model.fs = fs;
model.times = times;
model.freqs = freqs;
model.chans = chans;
model.baseModels = {};
model.configs = {};
model.useDV = p.Results.UseDecisionValues;
model.classifierType = p.Results.ClassifierType;
model.optimize = p.Results.Optimize;
model.timeLimit = p.Results.OptimizeTimeLimit;
model.verbose = p.Results.Verbose;
useParallel = p.Results.UseParallel;

% 计算样本数和子模型数量
nTrials = size(Xt, 3);
nTimes = max(size(times, 1),1);
nFreqs = size(freqs, 1);
nChans = max(numel(chans),1);
nSubModels = nTimes * nFreqs * nChans;

verbose = model.verbose;
if verbose
    fprintf('开始训练Stacking模型\n');
    fprintf('配置组合数: %d\n', nSubModels);
    if useParallel
        fprintf('使用并行计算加速\n');
    end
end

% 创建所有配置组合
allConfigs = cell(nSubModels, 1);
idx = 1;
for t = 1:nTimes
    time_win = times(t, :);
    for f = 1:nFreqs
        freq_band = freqs(f, :);
        for c = 1:nChans
            chan_idx = chans{c};
            allConfigs{idx} = struct(...
                'time_win', time_win, ...
                'freq_band', freq_band, ...
                'chan_idx', chan_idx, ...
                'index', idx);
            idx = idx + 1;
        end
    end
end

% 预分配子模型和配置存储
baseModels = cell(nSubModels, 1);
metaFeatures = zeros(nTrials, nSubModels);

% 根据是否使用并行计算选择循环类型
useDV = model.useDV;
if useParallel
    % 确保并行池已开启
    if isempty(gcp('nocreate'))
        parpool('local');
    end
    
    % 并行处理所有配置
    parfor i = 1:nSubModels
        config = allConfigs{i};
        
        if verbose
            fprintf('训练子模型 %d/%d: 时间窗[%.1f-%.1f]s, ', i, nSubModels, config.time_win(1), config.time_win(2));
            fprintf('频带[%.1f-%.1f]Hz, ', config.freq_band(1), config.freq_band(2));
            fprintf('通道%d个, ', numel(config.chan_idx));
        end
        
        % 数据预处理：时频滤波和通道选择
        fXs = ERPs_Filter(Xs, config.freq_band, config.chan_idx, config.time_win, fs);
        fXt = ERPs_Filter(Xt, config.freq_band, config.chan_idx, config.time_win, fs);
        
        % 训练子模型
        subModel = single_rsfda_modeling(fXs, Ys, fXt, Yt);
        
        % 在训练集上测试子模型（得到预测标签和决策值）
        [pred, dv, ~] = single_rsfda_classify(subModel, fXt, Yt);
        
        % 保存结果
        baseModels{i} = subModel;
        
        % 收集元特征
        if useDV
            metaFeatures(:, i) = dv;
        else
            metaFeatures(:, i) = pred;
        end
    end
else
    % 串行处理所有配置
    for i = 1:nSubModels
        config = allConfigs{i};
        
        if verbose
            fprintf('训练子模型 %d/%d: 时间窗[%.1f-%.1f]s, ', i, nSubModels, config.time_win(1), config.time_win(2));
            fprintf('频带[%.1f-%.1f]Hz, ', config.freq_band(1), config.freq_band(2));
            fprintf('通道%d个, ', numel(config.chan_idx));
        end
        
        % 数据预处理：时频滤波和通道选择
        fXs = ERPs_Filter(Xs, config.freq_band, config.chan_idx, config.time_win, fs);
        fXt = ERPs_Filter(Xt, config.freq_band, config.chan_idx, config.time_win, fs);
        
        % 训练子模型
        subModel = single_rsfda_modeling(fXs, Ys, fXt, Yt);
        
        % 在训练集上测试子模型（得到预测标签和决策值）
        [pred, dv, ~] = single_rsfda_classify(subModel, fXt, Yt);
        
        % 保存结果
        baseModels{i} = subModel;
        
        % 收集元特征
        if useDV
            metaFeatures(:, i) = dv;
        else
            metaFeatures(:, i) = pred;
        end
    end
end

% 将临时结果转移到模型结构
model.baseModels = baseModels;
model.configs = allConfigs;

% 训练元分类器
if verbose
    fprintf('训练元分类器 (%s)...\n', model.classifierType);
end

Y = cat(1, Ys(:), Yt(:));
model.metaModel = train_classifier(metaFeatures, Y, model.classifierType, ...
        model.optimize, model.timeLimit);

if verbose
    fprintf('Stacking模型训练完成\n');
end
end