%% Mutual Information based Best Individual Feature（MIBIF）
%输入
%fea_train(m*n), features that needs to culculate MI.
%label_train(m*1),label that conresponds the features
%输出
%sort_tmp(n*2),the first dimension includes the Mutual information values，and 
%the second dimension incoude the index
function sort_tmp=all_MuI(fea_train,label_train)
n=size(fea_train,1);               
tmp=[];
for i=1:size(fea_train,2)
    MuI=calc_MuI(fea_train(:,i),label_train,n);
    tmp=[tmp;MuI i];                  
end
sort_tmp=sortrows(tmp,'descend');
