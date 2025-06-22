%% RSFDA 基础模型训练程序
% Author: LC Pan
% Date: May. 1, 2025
% [1] Lc Pan, et al. Cross-session motor imagery-electroencephalography
% decoding with Riemannian spatial filtering and domain adaptation[J].
% Journal of Biomedical Engineering, 2025, 42(2):272-279.

function model = single_rsfda_modeling(Xs, Ys, Xt, Yt, options)
if nargin < 5 || isempty(options)
    options = struct();
end
if ~isfield(options, 'method_mean'), options.method_mean = 'riemann'; end
if ~isfield(options, 'method_feasel'), options.method_feasel = 'MIBIF'; end
if ~isfield(options, 'maxFeaNum'), options.maxFeaNum = 30; end
if ~isfield(options, 'classifierType'), options.classifierType = 'SVM'; end
if ~isfield(options, 'optimize'), options.optimize = false; end
if ~isfield(options, 'timeLimit'), options.timeLimit = 30; end

type=unique(Ys);
model.type=type;

% 空间滤波
[Wrsf,Xt,Xs]=RSF(Xt,Yt,Xs);
model.Wrsf=Wrsf;

% 数据预对齐
method_mean = options.method_mean;
Cs = covariances(Xs);
Mrct = mean_covariances(Cs, method_mean);
Mrct = Mrct^(-1/2);
for s=1:size(Cs,3)
    Cs(:,:,s)=Mrct*Cs(:,:,s)*Mrct;
end

Ct = covariances(Xt);
Mrct = mean_covariances(Ct, method_mean);
Mrct = Mrct^(-1/2);
for s=1:size(Ct,3)
    Ct(:,:,s)=Mrct*Ct(:,:,s)*Mrct;
end

model.Mrct=Mrct;

% 提取切空间特征
MC = mean_covariances(Cs, method_mean);
Fs = Tangent_space(Cs,MC)';
MC = mean_covariances(Ct, method_mean);
Ft = Tangent_space(Ct,MC)';

model.MC=MC;

% 特征选择
[Ft,index]=FeaturesSelection(Ft,Yt,options.method_feasel,options.maxFeaNum);
Fs=Fs(:,index);

model.index=index;

% 特征对齐
[Zs, Zt, Wt] = MEKT_P(Fs', Ft', Ys, Yt, options);

model.Wt=Wt;

% 分类
Z = cat(1, Zs', Zt');
Y = cat(1, Ys(:), Yt(:));
classifier = train_classifier(Z, Y, options.classifierType, ...
    options.optimize, options.timeLimit);

model.classifierType=options.classifierType;
model.classifier=classifier;

end



