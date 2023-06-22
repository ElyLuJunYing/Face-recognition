
function [predictSex_label, accuracySex] = sex(dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

%训练集男上女下
dataFaceR_sex =[dataFaceR_maleExtra;dataFaceR_femaleExtra];
dataFaceR_sex = real(dataFaceR_sex);
[RsexRow,RsexCol] =size(dataFaceR_sex);

%测试集男上女下
dataFaceS_sex = [dataFaceS_maleExtra;dataFaceS_femaleExtra];
dataFaceS_sex = real(dataFaceS_sex);
[SsexRow,SsexCol] = size(dataFaceS_sex);

[maleRow,maleCol] = size(dataFaceR_maleExtra);
[femaleRow,femaleCol] = size(dataFaceR_femaleExtra);
[maleRowS,maleColS] = size(dataFaceS_maleExtra);
[femaleRowS,femaleColS] = size(dataFaceS_femaleExtra);

%零列
lable_sexR = zeros(RsexRow,1);
lable_sexS = zeros(SsexRow,1);

%监督
for i=1:RsexRow
    if(i<=maleRow)
       lable_sexR(i,1) = 1;     %男性的标签为1，女性的标签为0
    elseif(i>maleRow)
       lable_sexR(i,1) = 0;
    end 
   
end

for i=1:SsexRow
    if(i<=maleRowS)
       lable_sexS(i,1) = 1;     %男性的标签为1，女性的标签为0
    elseif(i>maleRowS)
        lable_sexS(i,1)=0;
    end 
end

%%=======================================贝叶斯========================================
  t1=cputime;
  model =fitcnb(dataFaceR_sex,lable_sexR);  
  predictSex_label   =  predict(model,dataFaceS_sex);  
  t2=cputime;
  t=t2-t1;               %分类器训练所用时间
  a=0;
  for i=1:SsexRow
    if(predictSex_label(i,1)==lable_sexS(i,1))
        a=a+1;
    end
 end
 accuracySex = a/SsexRow;
 
end


