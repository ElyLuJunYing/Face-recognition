%% 将图像转为 jpg 格式
clear; clc; fclose('all');
fileFolder = fullfile('./rawdata'); % 引号内是需要遍历的路径，填绝对路径，然后保存在fileFolder
dirOutput = dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name};

%% 开始转换
for ii = 3:length(fileNames)
    fid=fopen(['./rawdata/',num2str(fileNames{ii})]); % 文件路径（注：包含文件名）
    I = fread(fid);
    matrix = reshape(I, sqrt(length(I)), sqrt(length(I)));
    matrix = rot90(matrix,3); % 旋转图像
    if length(matrix(:)) > 128*128 % 将2张512x512的图像resize成128x128
        img = imresize(mat2gray(matrix),[128,128],'bilinear','AntiAliasing',false);
    else
        img = mat2gray(matrix);
    end
        imwrite(img, ['./dataset/jpgdata/',num2str(fileNames{ii}),'.jpg']); % 创建JPG文件
    % imagesc(reshape(I, 128, 128)'); 
    % colormap(gray(256));
    fclose(fid);
end

disp('转换完成')