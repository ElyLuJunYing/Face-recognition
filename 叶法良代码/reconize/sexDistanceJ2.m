function [dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra] = sexDistanceJ2(dataFaceR_male,dataFaceR_female,dataFaceS_male,dataFaceS_female,dimension)
%UNTITLED2 此处显示有关此函数的摘要
%{
    函数输入：各个类别的数据，特征提取的维数dimension
    函数的输出：经过特征提取后的各类别数据
    函数的功能：基于可分性依据J2进行特征的提取

%}
%   此处显示详细说明
[maleRow,maleCol] = size(dataFaceR_male);%获取dataFaceR_male的行数和列数
[femaleRow,femaleCol] = size(dataFaceR_female);%获取dataFaceR_female的行数和列数
%求类内的均值向量
%maleMean = mean(dataFaceR_male(:,1:maleCol+1)); %男性的均值向量
maleMean = mean(dataFaceR_male); %男性的均值向量
%femaleMean = mean(dataFaceR_female(:,1:femaleCol+1));%女性的均值向量
femaleMean = mean(dataFaceR_female);%女性的均值向量
%类别先验概率
maleP = 0.5; %男性先验概率
femaleP = 0.5;%女性先验概率
sexTotalMean = maleMean.*maleP + femaleMean.*femaleP;  %性别总体均值
%求类间离散度矩阵

   discreteSbMale = (maleMean - sexTotalMean)' * (maleMean - sexTotalMean);
  
 discreteSbFemale = (femaleMean - sexTotalMean)' * (femaleMean - sexTotalMean);

discreteSb = discreteSbMale.*maleP + discreteSbFemale.*femaleP;
%求类内离散度矩阵
maleDiscrete = zeros(100,100);
femaleDiscrete = zeros(100,100);
for i=1:maleRow
    maleDiscrete = maleDiscrete + (dataFaceR_male(i,:) - maleMean(1,:))' * (dataFaceR_male(i,:) - maleMean(1,:));  
   
end
maleDiscreteSw = (maleDiscrete.*maleP)./maleRow;     %计算男性类的类内的离散度

for i=1:femaleRow
    
    femaleDiscrete = femaleDiscrete + (dataFaceR_female(i,:) - femaleMean(1,:))' * (dataFaceR_female(i,:) - femaleMean(1,:));   
end
femaleDiscreteSw = (femaleDiscrete.*femaleP)./femaleRow; %计算女性类的类内的离散度
discreteSw = maleDiscreteSw + femaleDiscreteSw;%类内总的离散度

%求变换矩阵
transforMatrix = discreteSw\discreteSb;
%求变换矩阵的特征值瑜特征向量
[featureVec,eigenValMat] = eig(transforMatrix);  %featureVec：特征向量,eigenValMat:对角矩阵
eigenVal = diag(eigenValMat);    %取对角矩阵的元素，组成一列
%求特征值矩阵的行数和列数
[eigenValRow,eigenValCol] = size(eigenVal);
vec = zeros(eigenValRow,1);        %做为中间变量进行排序
Eig = zeros(1,1);                           %作为中间变量对特征值从大到小排序
%将特征值从大到小排序并跟着调制特征向量。
for i=1:eigenValRow
    k = eigenValRow - i;
    for j=1:k
        Eig = eigenVal(j,1);
        vec(:,1) = featureVec(:,j);
        if(eigenVal(j+1,1)>=Eig)
            eigenVal(j,1) = eigenVal(j+1,1);
            featureVec(:,j) = featureVec(:,j+1);
            eigenVal(j+1,1) = Eig;
            featureVec(:,j+1) = vec(:,1); 
        end    
    end
end
%获得降到dimension维数的变换矩阵
extraMatrix = featureVec(:,1:dimension);
% extraMatrix = featureVec(:,1:50);
for i=1:maleRow
    dataFaceR_maleExtra1 = dataFaceR_male(i,:)*extraMatrix;  
    dataFaceR_maleExtra(i,:) = dataFaceR_maleExtra1;     
end
for j=1:femaleRow
    dataFaceR_femaleExtra1 = dataFaceR_female(j,:)*extraMatrix;
    dataFaceR_femaleExtra(j,:) = dataFaceR_femaleExtra1;
end

%=================================================================================================================
[maleRowS,maleColS] = size(dataFaceS_male);%获取dataFaceR_male的行数和列数
[femaleRowS,femaleColS] = size(dataFaceS_female);%获取dataFaceR_female的行数和列数

%求类内的均值向量
maleMeanS = mean(dataFaceS_male); %男性的均值向量
femaleMeanS = mean(dataFaceS_female);%女性的均值向量

%类别先验概率
malePS = 0.5; %男性先验概率
femalePS = 0.5;%女性先验概率
sexTotalMeanS = maleMeanS.*malePS + femaleMeanS.*femalePS;  %性别总体均值
%求类间离散度矩阵
   discreteSbMaleS = (maleMeanS - sexTotalMeanS)' *(maleMeanS - sexTotalMeanS) ;

   discreteSbFemaleS = (femaleMeanS - sexTotalMeanS)' *(femaleMeanS - sexTotalMeanS);

discreteSbS = discreteSbMaleS.*malePS + discreteSbFemaleS.*femalePS;
%求类内离散度矩阵
maleDiscreteS = zeros(100,100);
femaleDiscreteS = zeros(100,100);

for i=1:maleRowS  
    maleDiscreteS = maleDiscreteS + (dataFaceS_male(i,:) - maleMeanS(1,:))'*(dataFaceS_male(i,:) - maleMeanS(1,:));  
end
maleDiscreteSwS = (maleDiscreteS.*malePS)./maleRowS;     %计算男性类的类内的离散度
for i=1:femaleRowS
    femaleDiscreteS = femaleDiscreteS  + (dataFaceS_female(i,:) - femaleMeanS(1,:))' * (dataFaceS_female(i,:) - femaleMeanS(1,:));  
end
femaleDiscreteSwS = (femaleDiscreteS.*femalePS)./femaleRowS; %计算女性类的类内的离散度
discreteSwS = maleDiscreteSwS + femaleDiscreteSwS;%类内总的离散度
%求变换矩阵
% ivdiscreteSwS = inv(discreteSwS);             %Sw矩阵的逆
% transforMatrixS = ivdiscreteSwS .* discreteSbS;
%求变换矩阵
transforMatrixS = discreteSwS\discreteSbS;
%求变换矩阵的特征值瑜特征向量
[featureVecS,eigenValMatS] = eig(transforMatrixS);  %featureVec：特征向量,eigenValMat:对角矩阵
eigenValS = diag(eigenValMatS);    %取对角矩阵的元素，组成一列
%求特征值矩阵的行数和列数
[eigenValRowS,eigenValColS] = size(eigenValS);
vecS = zeros(eigenValRowS,1);        %做为中间变量进行排序
EigS = 0;                           %作为中间变量对特征值从大到小排序
%将特征值从大到小排序并跟着调制特征向量。
for i=1:eigenValRowS
    for j=1:eigenValRowS - i
        EigS = eigenValS(j,1);
        vecS(:,1) = featureVecS(:,j);
        if(eigenValS(j+1,1)>=EigS)
            eigenValS(j,1) = eigenValS(j+1,1);
            featureVecS(:,j) = featureVecS(:,j+1);
            eigenValS(j+1,1) = EigS;
            featureVecS(:,j+1) = vecS(:,1); 
        end    
    end
end
%获得降到dimension维数的变换矩阵
 extraMatrixS = featureVecS(:,1:dimension);
% extraMatrixS = featureVecS(:,1:50);

for i=1:maleRowS               
    dataFaceS_maleExtra1 =dataFaceS_male(i,:) * extraMatrixS;  
    dataFaceS_maleExtra(i,:) = dataFaceS_maleExtra1;      
end
for j=1:femaleRowS
    dataFaceS_femaleExtra1 =dataFaceS_female(j,:) * extraMatrixS;  
    dataFaceS_femaleExtra(j,:) = dataFaceS_femaleExtra1;     
end




end

