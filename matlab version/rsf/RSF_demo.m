%% Spatial filters based on Riemannian geometry
%  Author: Pan Lincong
%  Edition date: 11 Sep 2023
%  Lincong Pan, et al. Enhancing Motor Imagery EEG Classification with a 
%  Riemannian Geometry-Based Spatial Filtering (RSF) Method [J]. Neural 
%  Networks, 107511.

function [trainData,testData,dd,W]=RSF_demo(traindata,trainlabel,testdata,dim,method)
if ~exist('dim','var') || isempty(dim)
    dim = 4;
end
if ~exist('method','var') || isempty(method)
    method = 'default';
end

if ~strcmpi(method,'none')
    metric='riemann';
    labeltype = unique(trainlabel);
    traincov=covariances(traindata);
    covm1 = mean_covariances(traincov(:,:,trainlabel==labeltype(1)), metric);
    covm2 = mean_covariances(traincov(:,:,trainlabel==labeltype(2)), metric);
end

try
    switch lower(method)
        case 'none'
            trainData = traindata;
            W = eye(size(traindata,1));
            if ~exist('testdata','var') || isempty(testdata)
                testData = [];
            else
                testData = testdata;
            end
            %         dd=distance(covm1, covm2,'riemann');
            dd=NaN;
            return
        case 'csp'
            csp_filter=CSP(traincov,'label',trainlabel,'m',dim);
            W0=cat(2,csp_filter{1},csp_filter{2});
            W = optimizeRiemann2(covm1, covm2, [], W0);
        case 'riemann-csp'
            csp_filter=CSP(cat(3,covm1,covm2),'label',[1;2],'m',dim);
            W0=cat(2,csp_filter{1},csp_filter{2});
            W = optimizeRiemann2(covm1, covm2, [], W0);
        case 'csp2'
            csp_filter=CSP(traincov,'label',trainlabel,'m',dim);
            W1 = optimizeRiemann2(covm1, covm2, [], csp_filter{1});
            W2 = optimizeRiemann2(covm1, covm2, [], csp_filter{2});
            W = cat(2, W1, W2);
        case 'riemann-csp2'
            csp_filter=CSP(cat(3,covm1,covm2),'label',[1;2],'m',dim);
            W1 = optimizeRiemann2(covm1, covm2, [], csp_filter{1});
            W2 = optimizeRiemann2(covm1, covm2, [], csp_filter{2});
            W = cat(2, W1, W2);
        case 'cspf'
            csp_filter=CSP(traincov,'label',trainlabel,'m',dim);
            W=cat(2,csp_filter{1},csp_filter{2});
        case 'all'
            %csp
            csp_filter=CSP(traincov,'label',trainlabel,'m',dim);
            W0=cat(2,csp_filter{1},csp_filter{2});
            W_1 = optimizeRiemann2(covm1, covm2, [], W0);
            %csp2
            W_11 = optimizeRiemann2(covm1, covm2, [], csp_filter{1});
            W_12 = optimizeRiemann2(covm1, covm2, [], csp_filter{2});
            W_2 = cat(2, W_11, W_12);
            %riemann-csp
            csp_filter=CSP(cat(3,covm1,covm2),'label',[1;2],'m',dim);
            W0=cat(2,csp_filter{1},csp_filter{2});
            W_3 = optimizeRiemann2(covm1, covm2, [], W0);
            %riemann-csp2
            W_21 = optimizeRiemann2(covm1, covm2, [], csp_filter{1});
            W_22 = optimizeRiemann2(covm1, covm2, [], csp_filter{2});
            W_4 = cat(2, W_21, W_22);

            d1 = distance(W_1'*covm1*W_1, W_1'*covm2*W_1,'riemann');
            d2 = distance(W_2'*covm1*W_2, W_2'*covm2*W_2,'riemann');
            d3 = distance(W_3'*covm1*W_3, W_3'*covm2*W_3,'riemann');
            d4 = distance(W_4'*covm1*W_4, W_4'*covm2*W_4,'riemann');
            [~,ind] = max([d1,d2,d3,d4]);
            W = eval(['W_',num2str(ind)]);
        otherwise
            W = optimizeRiemann2(covm1, covm2, dim*2);
    end
catch ME
    save_error(ME,'errorLog.txt',false,'dim',dim,'method',method)
    [trainData,testData,dd,W]=RSF(traindata,trainlabel,testdata,dim);
    return
end


trainData=zeros(dim*2, size(traindata,2), size(traindata,3));
for i=1:size(trainData,3)
    trainData(:,:,i)=W'*traindata(:,:,i);
end

if ~exist('testdata','var') || isempty(testdata)
    testData = [];
else
    testData=zeros(dim*2, size(testdata,2), size(testdata,3));
    for i=1:size(testData,3)
        testData(:,:,i)=W'*testdata(:,:,i);
    end
end

dd=distance(W'*covm1*W, W'*covm2*W,'riemann');
    
end


%% **************************************************************************
%% 基于黎曼几何的空间滤波方法(带约束)
function W_opt = optimizeRiemann(P1, P2, N, W0, maxiter)
if ~exist('maxiter','var') || isempty(maxiter)
    % 设置初始投影矩阵 W
    maxiter = 10000;
end
if ~exist('W0','var') || isempty(W0)
    % 设置初始投影矩阵 W
    M = size(P1, 1);
    W0 = randn(M, N);
    W0 = orth(W0);
end

% 定义目标函数
% objFunc = @(W) -norm(logm(pinv(W) * inv(P1) * P2 * W), 'fro');
objFunc = @(W) -sum(log(eig(W'*P1*W, W'*P2*W)).^2);

% 定义约束条件
W_constraint = @(W) deal([], W' * W - eye(N));

% 定义优化器
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter');
% options = optimoptions('fmincon', 'Algorithm', 'interior-point');
options.MaxFunctionEvaluations = maxiter; % 增加最大函数评估次数限制

% 开始优化计算
W1 = fmincon(objFunc, W0, [], [], [], [], [], [], W_constraint, options);

d0 = sum(log(eig(W0'*P1*W0, W0'*P2*W0)).^2);
d1 = sum(log(eig(W1'*P1*W1, W1'*P2*W1)).^2);
if d0>d1
    W_opt = W0;
else
    W_opt = W1;
end
end

%% 基于黎曼几何的空间滤波方法(无约束)
function W_opt = optimizeRiemann2(P1, P2, N, W0, maxiter)
if ~exist('maxiter','var') || isempty(maxiter)
    % 设置初始投影矩阵 W
    maxiter = 15000;
end
if ~exist('W0','var') || isempty(W0)
    % 设置初始投影矩阵 W
    M = size(P1, 1);
    W0 = randn(M, N);
    W0 = orth(W0);
end

% 定义目标函数
% objFunc = @(W) -norm(logm(pinv(W) * inv(P1) * P2 * W), 'fro');
objFunc = @(W) -sum(log(eig(W'*P1*W, W'*P2*W)).^2);

% 定义优化器
% options = optimoptions('fminunc', 'Display', 'iter','PlotFcn','optimplotfval');
options = optimoptions('fminunc');
options.MaxFunctionEvaluations = maxiter; % 增加最大函数评估次数限制

% 开始优化计算
W1 = fminunc(objFunc, W0, options);

d0 = sum(log(eig(W0'*P1*W0, W0'*P2*W0)).^2);
d1 = sum(log(eig(W1'*P1*W1, W1'*P2*W1)).^2);
if d0>d1
    W_opt = W0;
else
    W_opt = W1;
end
end

%% 基于黎曼几何的投影方法(无约束)
function W_opt = optimizeRiemann3(P1, P2, W0, maxiter)
if ~exist('maxiter','var') || isempty(maxiter)
    % 设置初始投影矩阵 W
    maxiter = 15000;
end
if ~exist('W0','var') || isempty(W0)
    % 设置初始投影矩阵 W
    N = size(P1, 1);
    W0 = eye(N);
end

% 定义目标函数
objFunc = @(W) -sum(log(eig(P1*W, P2*W)).^2);

% 定义优化器
options = optimoptions('fminunc');
options.MaxFunctionEvaluations = maxiter; % 增加最大函数评估次数限制

% 开始优化计算
W_opt = fminunc(objFunc, W0, options);
end