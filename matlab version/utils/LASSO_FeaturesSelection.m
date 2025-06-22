%% 基于LASSO回归的特征选择
% 固定输入
% X: 特征集, 2D矩阵:样本数*特征维度
% y: 标签值, 向量，长度为样本数
% 可选输入
% maxNum: 保留的特征维度最大数量
% alpha: 取值范围是[0,1],当alpha等于1时,表示套索回归;当alpha趋近于0时,表示岭回归;当alpha介于0与1之间时,表示弹性网络.
% 输出
% Xselect: 经过LASSO回归后系数不为0的特征维度构成的特征集
% selectedFeatures: 选择的特征索引
% coefMinMSE: 所有维度特征的LASSO回归系数

function [Xselect,selectedFeatures]=LASSO_FeaturesSelection(X,y,maxNum,alpha)
if nargin<4
    %默认选择套索回归
    alpha=1;
end
if alpha == 0
    alpha=0.0001;
end
if nargin<3 || isempty(maxNum)
    %默认设置特征选择数量至少比原特征维度少1维度
    maxNum=size(X,2)-1;
end

% 1.加载或创建预测值数据矩阵X和回应向量y
% X = randn(120,6);
% weights = [0.5;2;-0.01;-3;0;0.0002]; % Only two nonzero coefficients
% y = X*weights + randn(120,1)*0.1; % Small added noise

% 2. 使用套索函数进行套索正则化，得到正则化参数lambda的不同值的拟合系数。您还可以
% 指定其他选项，例如交叉验证折叠的数量、弹性网的Alpha值和预测值的标准化。
% 当'Alpha'设置为1时，表示套索回归；当'Alpha'趋近于0时，表示岭回归；当'Alpha'介于0与1之间时，表示弹性网络。
[B,FitInfo] = lasso(X,y,'CV',5,'Alpha',alpha,'Standardize',true,'DFmax',maxNum);

% 3. 根据最小交叉验证均方误差(MSE)或最小均方误差加1个标准误差等准则选择最优的lambda
% 值。您可以使用FitInfo结构访问这些值。

% lambdaMinMSE = FitInfo.LambdaMinMSE;
% lambda1SE = FitInfo.Lambda1SE;

% 4. 以最优lambda值为指标，从B中提取相应的系数向量。

idxMinMSE = FitInfo.IndexMinMSE;
coefMinMSE = B(:,idxMinMSE);

% 5. 识别所选向量中的非零系数，其指示所选特征。您可以使用Find函数来获取它们的索引。

selectedFeatures = find(coefMinMSE);

% 6. 使用所选特征来创建简化的预测器矩阵，并使用Fitlm函数来拟合线性模型。您还可以
% 使用预测功能，通过拟合的模型对新数据进行预测。

Xselect = X(:,selectedFeatures);

% mdl = fitlm(Xreduced,y);
% ypred = predict(mdl,Xnew(:,selectedFeatures));
end
