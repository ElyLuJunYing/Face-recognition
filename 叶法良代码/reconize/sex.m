
function [predictSex_label, accuracySex] = sex(dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

%ѵ��������Ů��
dataFaceR_sex =[dataFaceR_maleExtra;dataFaceR_femaleExtra];
dataFaceR_sex = real(dataFaceR_sex);
[RsexRow,RsexCol] =size(dataFaceR_sex);

%���Լ�����Ů��
dataFaceS_sex = [dataFaceS_maleExtra;dataFaceS_femaleExtra];
dataFaceS_sex = real(dataFaceS_sex);
[SsexRow,SsexCol] = size(dataFaceS_sex);

[maleRow,maleCol] = size(dataFaceR_maleExtra);
[femaleRow,femaleCol] = size(dataFaceR_femaleExtra);
[maleRowS,maleColS] = size(dataFaceS_maleExtra);
[femaleRowS,femaleColS] = size(dataFaceS_femaleExtra);

%����
lable_sexR = zeros(RsexRow,1);
lable_sexS = zeros(SsexRow,1);

%�ල
for i=1:RsexRow
    if(i<=maleRow)
       lable_sexR(i,1) = 1;     %���Եı�ǩΪ1��Ů�Եı�ǩΪ0
    elseif(i>maleRow)
       lable_sexR(i,1) = 0;
    end 
   
end

for i=1:SsexRow
    if(i<=maleRowS)
       lable_sexS(i,1) = 1;     %���Եı�ǩΪ1��Ů�Եı�ǩΪ0
    elseif(i>maleRowS)
        lable_sexS(i,1)=0;
    end 
end

%%=======================================��Ҷ˹========================================
  t1=cputime;
  model =fitcnb(dataFaceR_sex,lable_sexR);  
  predictSex_label   =  predict(model,dataFaceS_sex);  
  t2=cputime;
  t=t2-t1;               %������ѵ������ʱ��
  a=0;
  for i=1:SsexRow
    if(predictSex_label(i,1)==lable_sexS(i,1))
        a=a+1;
    end
 end
 accuracySex = a/SsexRow;
 
end


