%% CSP/RCSP算法
%  Author: Pan Lincong
%  Edition date: 1 May 2019

%输入
% data:数据集
%可选输入
% label,输出CSP滤波器
% csp_filter，输出CSP/RCSP提取的特征
% otherdata;otherlabel，输出RCSP滤波器

%eg.
%csp_filter=CSP(AtrainData,'label',trainLabel);
%rcsp_filter=CSP(AtrainData,'label',trainLabel,'otherdata',otherdata,'otherlabel',otherlabel);
%

function output=CSP(data,p1,v1,p2,v2,p3,v3)
% feature=[];
csp_filter=[];
if nargin < 2
    error('CSP函数输入参数不够！')
else
    for i = 2:2:nargin
        Param = eval(['p',int2str((i-2)/2 +1)]);
        Value = eval(['v',int2str((i-2)/2 +1)]);
        if ~isstr(Param)
            error('Flag arguments must be strings')
        end
        Param = lower(Param);
        switch Param
            case 'label'
                label=Value;
            case 'csp_filter'
                csp_filter=Value;
            case 'otherdata'
                otherdata=Value;
            case 'otherlabel'
                otherlabel=Value;
            case 'm'
                m=Value;
        end
    end
    if ~exist('m','var')
        m=4;%单个类别的特征维度
    end
    if exist('otherdata','var') && exist('otherlabel','var')
        csp_filter=p_rcspmulticlass(data,otherdata,label,otherlabel,m);
        output=csp_filter;
    elseif exist('label','var')
        csp_filter=p_cspmulticlass(data,label,m);
%       csp_filter=q_cspmulticlass(data,label);
        output=csp_filter;
    elseif exist('csp_filter','var')
        feature=p_cspfeature(data,csp_filter);
%       feature=q_cspfeature(csp_filter,data,m);
        output=feature;
    end
end

end

function csp_filter=p_cspmulticlass(data,label,m)

labclas = unique(label);

for k=1:length(labclas)
    
    ind=find(label==labclas(k));
    data1=data(:,:,ind);
    if issymmetric(mean(data1,3))
        covm_temp=data1;
    else
        for i=1:length(ind)
            temp=data1(:,:,i);  %选取某一试次样本            
%             covm_temp(:,:,i)=(temp*temp')/trace(temp*temp');% 求协方差矩阵     
            covm_temp(:,:,i)=(temp*temp')/(size(temp,2)-1);% 求协方差矩阵  
        end
    end
    covm_mean(:,:,k)=mean(covm_temp,3);%代表第k类的平均协方差矩阵
end
csp_filter=csp_feature_extraction(covm_mean,m); % 经典CSP

end

function csp_filter=p_rcspmulticlass(data0,data1,label0,label1,m)
beta=0.1;gamma=0.1;
labclas = unique(label0);
for k=1:length(labclas)
    covm_mean(:,:,k)=RCSP_Cov(data0(:,:,label0==labclas(k)),data1(:,:,label1==labclas(k)),beta,gamma);
end
csp_filter=csp_feature_extraction(covm_mean,m); % RCSP
end

function csp_filter=csp_feature_extraction(covm_mean,m)
for k=1:size(covm_mean,3)
    
    indtemp=[1:size(covm_mean,3)]; indtemp(k)=[];
    X=covm_mean(:,:,k);
    Y=covm_mean(:,:,indtemp(1));
    if length(indtemp)==1
        Z=zeros(size(Y));
    elseif length(indtemp)==2
        Z=covm_mean(:,:,indtemp(2));
        M2= Y\Z;[V2, D2] = eig(M2);D2=diag(D2);[~, egIndex] = sort(D2,'descend','ComparisonMethod','abs');V2 = V2(:,egIndex);
    end
        
%     M1= (Y+Z)\X;[V1, D1] = eig(M1);D1=diag(D1);[~, egIndex] = sort(D1,'descend','ComparisonMethod','abs');V1 = V1(:,egIndex);
%     M4= X\(Y+Z);[V4, D4] = eig(M4);D4=diag(D4);[~, egIndex] = sort(D4,'descend','ComparisonMethod','abs');V4 = V4(:,egIndex);

    M1= Y\X;[V1, D1] = eig(M1);D1=diag(D1);[~, egIndex] = sort(D1,'descend','ComparisonMethod','abs');V1 = V1(:,egIndex);
    csp_filter{k} = V1(:,1:m);

%     csp_filter{k} = [V1(:,1:m) V2(:,1:m)];
end
end

function feature=p_cspfeature(data,csp_filter)

num_samp=size(data,3);
num_clas=length(csp_filter);
num_comp=size(csp_filter{1},2);
feature=zeros(num_samp,num_clas*num_comp);
for i=1:num_samp
    temp=data(:,:,i);
    for j=1:num_clas
        if issymmetric(temp)
            feature(i,(j-1)*num_comp+1:j*num_comp)=log(diag(csp_filter{j}'*temp*csp_filter{j}));
        else
            filtertemp=csp_filter{j}'*temp;
            feature(i,(j-1)*num_comp+1:j*num_comp)=log(diag(filtertemp * filtertemp'));
        end
    end
end
end