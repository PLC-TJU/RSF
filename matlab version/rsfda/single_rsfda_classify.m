%% RSFDA 基础模型分类程序
% Author: LC Pan
% Date: May. 1, 2025
% [1] Lc Pan, et al. Cross-session motor imagery-electroencephalography
% decoding with Riemannian spatial filtering and domain adaptation[J].
% Journal of Biomedical Engineering, 2025, 42(2):272-279.

function [predlabel, decision_values, testacc] = single_rsfda_classify(model, testdata, testlabel)
if nargin < 3
    testlabel = [];
end

% 空间滤波
Wrsf=model.Wrsf;
ftestdata=nan(size(Wrsf,2),size(testdata,2),size(testdata,3));
for s=1:size(testdata,3)
    ftestdata(:,:,s)=Wrsf'*testdata(:,:,s);
end

% 数据预对齐
Mrct=model.Mrct;
C = covariances(ftestdata);
for s=1:size(C,3)
    C(:,:,s)=Mrct*C(:,:,s)*Mrct;
end

% 提取切空间特征
MC=model.MC;
F = Tangent_space(C,MC)';

% 特征选择
index=model.index;
F=F(:,index);

% 特征对齐
Wt=model.Wt;
Z = Wt*F';

% 分类
classifier=model.classifier;
[predlabel, scores] = predict(classifier, Z');
decision_values = scores(:, 2);  % 正类概率

% 计算准确率
if ~isempty(testlabel)
    testacc = mean(predlabel == testlabel) * 100;
else
    testacc = [];
end

end