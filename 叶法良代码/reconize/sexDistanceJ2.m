function [dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra] = sexDistanceJ2(dataFaceR_male,dataFaceR_female,dataFaceS_male,dataFaceS_female,dimension)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%{
    �������룺�����������ݣ�������ȡ��ά��dimension
    ���������������������ȡ��ĸ��������
    �����Ĺ��ܣ����ڿɷ�������J2������������ȡ

%}
%   �˴���ʾ��ϸ˵��
[maleRow,maleCol] = size(dataFaceR_male);%��ȡdataFaceR_male������������
[femaleRow,femaleCol] = size(dataFaceR_female);%��ȡdataFaceR_female������������
%�����ڵľ�ֵ����
%maleMean = mean(dataFaceR_male(:,1:maleCol+1)); %���Եľ�ֵ����
maleMean = mean(dataFaceR_male); %���Եľ�ֵ����
%femaleMean = mean(dataFaceR_female(:,1:femaleCol+1));%Ů�Եľ�ֵ����
femaleMean = mean(dataFaceR_female);%Ů�Եľ�ֵ����
%����������
maleP = 0.5; %�����������
femaleP = 0.5;%Ů���������
sexTotalMean = maleMean.*maleP + femaleMean.*femaleP;  %�Ա������ֵ
%�������ɢ�Ⱦ���

   discreteSbMale = (maleMean - sexTotalMean)' * (maleMean - sexTotalMean);
  
 discreteSbFemale = (femaleMean - sexTotalMean)' * (femaleMean - sexTotalMean);

discreteSb = discreteSbMale.*maleP + discreteSbFemale.*femaleP;
%��������ɢ�Ⱦ���
maleDiscrete = zeros(100,100);
femaleDiscrete = zeros(100,100);
for i=1:maleRow
    maleDiscrete = maleDiscrete + (dataFaceR_male(i,:) - maleMean(1,:))' * (dataFaceR_male(i,:) - maleMean(1,:));  
   
end
maleDiscreteSw = (maleDiscrete.*maleP)./maleRow;     %��������������ڵ���ɢ��

for i=1:femaleRow
    
    femaleDiscrete = femaleDiscrete + (dataFaceR_female(i,:) - femaleMean(1,:))' * (dataFaceR_female(i,:) - femaleMean(1,:));   
end
femaleDiscreteSw = (femaleDiscrete.*femaleP)./femaleRow; %����Ů��������ڵ���ɢ��
discreteSw = maleDiscreteSw + femaleDiscreteSw;%�����ܵ���ɢ��

%��任����
transforMatrix = discreteSw\discreteSb;
%��任���������ֵ���������
[featureVec,eigenValMat] = eig(transforMatrix);  %featureVec����������,eigenValMat:�ԽǾ���
eigenVal = diag(eigenValMat);    %ȡ�ԽǾ����Ԫ�أ����һ��
%������ֵ���������������
[eigenValRow,eigenValCol] = size(eigenVal);
vec = zeros(eigenValRow,1);        %��Ϊ�м������������
Eig = zeros(1,1);                           %��Ϊ�м����������ֵ�Ӵ�С����
%������ֵ�Ӵ�С���򲢸��ŵ�������������
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
%��ý���dimensionά���ı任����
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
[maleRowS,maleColS] = size(dataFaceS_male);%��ȡdataFaceR_male������������
[femaleRowS,femaleColS] = size(dataFaceS_female);%��ȡdataFaceR_female������������

%�����ڵľ�ֵ����
maleMeanS = mean(dataFaceS_male); %���Եľ�ֵ����
femaleMeanS = mean(dataFaceS_female);%Ů�Եľ�ֵ����

%����������
malePS = 0.5; %�����������
femalePS = 0.5;%Ů���������
sexTotalMeanS = maleMeanS.*malePS + femaleMeanS.*femalePS;  %�Ա������ֵ
%�������ɢ�Ⱦ���
   discreteSbMaleS = (maleMeanS - sexTotalMeanS)' *(maleMeanS - sexTotalMeanS) ;

   discreteSbFemaleS = (femaleMeanS - sexTotalMeanS)' *(femaleMeanS - sexTotalMeanS);

discreteSbS = discreteSbMaleS.*malePS + discreteSbFemaleS.*femalePS;
%��������ɢ�Ⱦ���
maleDiscreteS = zeros(100,100);
femaleDiscreteS = zeros(100,100);

for i=1:maleRowS  
    maleDiscreteS = maleDiscreteS + (dataFaceS_male(i,:) - maleMeanS(1,:))'*(dataFaceS_male(i,:) - maleMeanS(1,:));  
end
maleDiscreteSwS = (maleDiscreteS.*malePS)./maleRowS;     %��������������ڵ���ɢ��
for i=1:femaleRowS
    femaleDiscreteS = femaleDiscreteS  + (dataFaceS_female(i,:) - femaleMeanS(1,:))' * (dataFaceS_female(i,:) - femaleMeanS(1,:));  
end
femaleDiscreteSwS = (femaleDiscreteS.*femalePS)./femaleRowS; %����Ů��������ڵ���ɢ��
discreteSwS = maleDiscreteSwS + femaleDiscreteSwS;%�����ܵ���ɢ��
%��任����
% ivdiscreteSwS = inv(discreteSwS);             %Sw�������
% transforMatrixS = ivdiscreteSwS .* discreteSbS;
%��任����
transforMatrixS = discreteSwS\discreteSbS;
%��任���������ֵ���������
[featureVecS,eigenValMatS] = eig(transforMatrixS);  %featureVec����������,eigenValMat:�ԽǾ���
eigenValS = diag(eigenValMatS);    %ȡ�ԽǾ����Ԫ�أ����һ��
%������ֵ���������������
[eigenValRowS,eigenValColS] = size(eigenValS);
vecS = zeros(eigenValRowS,1);        %��Ϊ�м������������
EigS = 0;                           %��Ϊ�м����������ֵ�Ӵ�С����
%������ֵ�Ӵ�С���򲢸��ŵ�������������
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
%��ý���dimensionά���ı任����
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

