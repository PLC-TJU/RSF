% 分类器
% Author: LC Pan
% Date: Jul. 1, 2024

function classifier = train_classifier(X, y, classifierType, optimize, timeLimit)
if nargin<5 || isempty(timeLimit)
    timeLimit=30;
end
if nargin<4 || isempty(optimize)
    optimize=false;
end
if nargin<3 || isempty(classifierType)
    classifierType='SVM';
end

if optimize
    switch upper(classifierType)
        case 'SVM'
            classifier = fitcsvm(X, y, ...
                'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
                'HyperparameterOptimizationOptions', struct(...
                'MaxTime', timeLimit, ...
                'ShowPlots', false, ...
                'Verbose', 0));
        case 'LOGISTIC'
            classifier = fitclinear(X, y, ...
                'Learner', 'logistic', ...
                'OptimizeHyperparameters', {'Lambda'}, ...
                'HyperparameterOptimizationOptions', struct(...
                'MaxTime', timeLimit, ...
                'ShowPlots', false, ...
                'Verbose', 0));
        case 'LDA'
            classifier = fitcdiscr(X, y, ...
                'OptimizeHyperparameters', {'Gamma', 'Delta'}, ...
                'HyperparameterOptimizationOptions', struct(...
                'MaxTime', timeLimit, ...
                'ShowPlots', false, ...
                'Verbose', 0));
        otherwise
            classifier = train_classifier(X, y, classifierType, false, 0);
    end
else
    switch upper(classifierType)
        case 'SVM'
            classifier = fitcsvm(X, y, 'KernelFunction', 'linear');
        case 'LOGISTIC'
            classifier = fitclinear(X, y, 'Learner', 'logistic');
        case 'LDA'
            classifier = fitcdiscr(X, y, 'DiscrimType', 'linear');
        case 'LIBSVM'
            classifier = libsvmtrain(y, X,'-t 0 -c 1 -q'); %需要安装libsvm
        otherwise
            error('未知分类器类型: %s', classifierType);
    end
end
end