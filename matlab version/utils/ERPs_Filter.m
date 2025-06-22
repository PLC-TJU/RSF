%% 样本数据预处理
%  Author: LC.Pan
%  Edition date: 22 April 2023

function Data=ERPs_Filter(data,freqs,channel,timewindow,fs,filterorder,filterflag)
%input:data:channels*points*samples
%滤波器参数
if nargin < 7
    filterflag = 'filtfilt';
end
if nargin < 6 || isempty(filterorder)
    filterorder = 5;   
end
if nargin < 5 || isempty(fs)
    fs=250;
end
if nargin < 4
    timewindow=[];
end
if nargin < 3
    channel=[];
end
if nargin < 2
    error('ERPs_Filter函数的输入参数不够！');
end

% 确保输入样本为：通道数*时间点数*样本数
if ismatrix(data)
    data = reshape(data, size(data,1), size(data,2), 1);
end

%通道选择
channel=sort(channel,'ascend');
if ~isempty(channel) 
    if channel(end)>size(data,1)
        warning('所选导联超出数据限制，已取消导联筛选！')
    end
    data=data(channel,:,:);
end

%时间窗截取
if ~isempty(timewindow)
    if length(timewindow)==1
        timewindow=[0,timewindow];
    end
    if timewindow(2)*fs>size(data,2)
        timewindow(2)=round(size(data,2)/fs,1);
        warning(['所选时间窗超出数据限制，已自动调整时间窗为:',num2str(timewindow(1)),'-',num2str(timewindow(2)),'s'])
    end
    data=data(:,round(timewindow(1)* fs) + 1:round(timewindow(2) * fs),:);
end

%CAR共参考平均
% data=data-mean(data);

%去基线漂移
data1=zeros(size(data,2),size(data,1),size(data,3));
for s=1:size(data,3)
    data1(:,:,s)=detrend(data(:,:,s)'); 
end

%去工频
data2 = Notch50Hz(data1);

%带通滤波
if iscell(freqs)
    f_a=freqs{1};
    f_b=freqs{2};
else
    filtercutoff = [2*freqs(1)/fs 2*freqs(2)/fs];
    [f_b, f_a] = butter(filterorder,filtercutoff);
end

switch filterflag
    case 'filter'
        data3 = filter(f_b,f_a,data2);
    case 'filtfilt'
        data3 = filtfilt(f_b,f_a,data2);
end

Data = permute(data3,[2,1,3]);

end

%% 50Hz陷波
function data = Notch50Hz(data)
% data:N_times * N_channels * N_trials
fs=250;
Fo = 50;
Q = 35;
BW = (Fo/(fs/2))/Q;
[B,A] = iircomb(fs/Fo,BW,'notch');
data = filtfilt(B,A,data);
end