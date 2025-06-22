% Pan.LC 2024.4.6
% 函数功能：
% 该函数用于在Matlab环境中捕获错误信息，并将其保存到指定的文本文件中。
% 它还可以保存错误发生时的变量值，以便于后续的调试和分析。

% 输入参数：
% ME (必需): MException对象，包含了错误信息和堆栈信息。

% txtfile (可选): 字符串，指定保存错误日志的文件名。如果未提供或为空，
% 将默认使用'errorLog.txt'。

% flag (可选): 布尔值，指定在记录错误信息后是否重新抛出异常。默认为false。

% varargin (可选): 可变参数，允许用户传入任意数量的变量名称-值对，
% 格式为'变量名', 变量值。这些变量将被记录在错误日志中。

function save_error(ME, txtfile, flag, varargin)
    % 设置默认文件名
    if nargin < 2 || isempty(txtfile)
        txtfile = 'errorLog.txt';
    end
    % 设置默认标志
    if nargin < 3 || isempty(flag)
        flag = false;
    end

    % 打开文件以追加错误信息
    fid = fopen(txtfile, 'a+');
    if fid == -1
        error('无法打开文件。');
    end

    % 获取当前时间戳
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    % 写入时间戳和错误信息
    fprintf(fid, '\n\n%s\n', repmat('-', 1, 80)); % 添加分隔线
    fprintf(fid, '时间戳: %s\n', timestamp);
    fprintf(fid, '%s\n', ME.message);

    % 写入错误的堆栈信息
    for e = 1:length(ME.stack)
        fprintf(fid, '在 %s 的第 %i 行发生错误。\n', ME.stack(e).name, ME.stack(e).line);
    end

    % 写入额外的可选输入信息
    for i = 1:2:length(varargin)
        if i+1 <= length(varargin)
            varName = varargin{i};
            varValue = varargin{i+1};
            fprintf(fid, '变量 %s: %s\n', varName, mat2str(varValue));
        end
    end

    % 关闭文件
    fclose(fid);

    % 如果需要，可以重新抛出错误
    if flag
        rethrow(ME);
    end
end
