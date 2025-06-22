%% 特征选择
% Author: LC Pan
% Date: Jul. 1, 2024

% 固定输入
% Fea: 特征集，2D矩阵:样本数*特征维度
% label: 标签值, 列向量，长度为样本数
% 可选输入
% method: 特征选择方法['MIBIF'/'LASSO']
% maxFeaNum: 最大输出特征维度
% 输出
% FeaSelect: 筛选后的特征集
% index: 所选特征在原特征集中的索引
function [FeaSelect,index]=FeaturesSelection(Fea,label,method,maxFeaNum)
if nargin<4
    maxFeaNum=[];
end
if nargin<3
    method='MIBIF';
end

if size(Fea,2)<=3
    FeaSelect=Fea;
    index=1:size(Fea,2);
    warning('输入的特征维度过小，不适合进行特征选择！')
    return;
end

switch upper(method)
    case 'MIBIF'
        if isempty(maxFeaNum) || maxFeaNum>=size(Fea,2)
            if size(Fea,2)>100
                %默认保留约30%的特征数量
                maxFeaNum=round(0.3*size(Fea,2));
            elseif size(Fea,2)>50
                %默认保留约50%的特征数量
                maxFeaNum=round(0.5*size(Fea,2));
            else
                %默认保留约70%的特征数量
                maxFeaNum=round(0.7*size(Fea,2));
            end
        end
        sort_tmp=all_MuI(Fea,label);
        index=sort_tmp(1:maxFeaNum,2);
        FeaSelect=Fea(:,index);
    case 'LASSO'
        %默认保留的特征维数小于原特征维数
        [FeaSelect,index]=LASSO_FeaturesSelection(Fea,label,maxFeaNum);
end