%% RSFDA Stacking集成学习模型分类程序
% Author: LC Pan
% Date: May. 4, 2025
% [1] Lc Pan, et al. Cross-session motor imagery-electroencephalography
% decoding with Riemannian spatial filtering and domain adaptation[J].
% Journal of Biomedical Engineering, 2025, 42(2):272-279.


function [predlabel, decision_values, testacc] = rsfda_classify(model, testdata, testlabel)
% STACKING_CLASSIFY 使用Stacking模型进行分类
% 支持并行计算子模型预测

% 输入:
%   model: 训练好的Stacking模型
%   testdata: 测试数据 (通道×时间点×样本数)
%   testlabel: 测试标签 (可选)
% 输出:
%   predlabel: 预测标签
%   decision_values: 决策值
%   testacc: 测试准确率 (如果有测试标签)

if ~exist('testlabel', 'var')
    testlabel = [];
end

% 获取样本数
nTrials = size(testdata, 3);
nSubModels = numel(model.baseModels);

if model.verbose
    fprintf('使用Stacking模型进行分类\n');
    fprintf('子模型数: %d\n', nSubModels);
end

% 创建配置列表
configs = model.configs;

% 根据子模型数量决定是否使用并行
useParallel = (nSubModels > 10) && (isempty(gcp('nocreate')) == false);
if model.verbose && useParallel
    fprintf('使用并行计算加速子模型预测\n');
end

% 预分配结果
metaFeatures = zeros(nTrials, nSubModels);
baseModels = model.baseModels;
fs=model.fs;
useDV=model.useDV;
if useParallel
    % 并行处理所有子模型
    parfor i = 1:nSubModels
        subModel = baseModels{i};
        config = configs{i};
        
        % 数据预处理：时频滤波和通道选择
        fdata = ERPs_Filter(testdata, config.freq_band, config.chan_idx, config.time_win, fs);
        
        % 使用子模型分类
        [pred, dv] = single_rsfda_classify(subModel,fdata,testlabel);
        
        % 收集元特征
        if useDV
            metaFeatures(:, i) = dv;
        else
            metaFeatures(:, i) = pred;
        end
    end
else
    % 串行处理所有子模型
    for i = 1:nSubModels
        subModel = baseModels{i};
        config = configs{i};
        
        if model.verbose
            fprintf('处理子模型 %d/%d: 时间窗[%.1f-%.1f]s, ', i, nSubModels, config.time_win(1), config.time_win(2));
            fprintf('频带[%.1f-%.1f]Hz, ', config.freq_band(1), config.freq_band(2));
            fprintf('通道%d个, ', numel(config.chan_idx));
        end
        
        % 数据预处理：时频滤波和通道选择
        fdata = ERPs_Filter(testdata, config.freq_band, config.chan_idx, config.time_win, fs);

        % 使用子模型分类
        [pred, dv] = single_rsfda_classify(subModel,fdata,testlabel);
        
        % 收集元特征
        if useDV
            metaFeatures(:, i) = dv;
        else
            metaFeatures(:, i) = pred;
        end
    end
end

% 使用元分类器进行最终预测
[predlabel, scores] = predict(model.metaModel, metaFeatures);
decision_values = scores(:, 2);  % 正类概率

% 计算准确率
if ~isempty(testlabel)
    testacc = mean(predlabel == testlabel) * 100;
    if model.verbose
        fprintf('测试准确率: %.2f%%\n', testacc);
    end
else
    testacc = [];
end
end