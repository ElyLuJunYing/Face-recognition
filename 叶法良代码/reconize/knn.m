function [predictSex_labels, accuracySexs] = knn(dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra)

data = [dataFaceR_maleExtra;dataFaceS_maleExtra;dataFaceR_femaleExtra;dataFaceS_femaleExtra]; %RS�������
data = real(data);
[M,N]= size(data);   %���ݼ�Ϊһ��M*N�ľ�������ÿһ�д���һ������

datamale = [dataFaceR_maleExtra;dataFaceS_maleExtra];
datafemale = [dataFaceR_femaleExtra;dataFaceS_femaleExtra];
[datamaleRow,datamaleCol] = size(datamale); 
[datafemaleRow,datafemaleCol] = size(datafemale);

%����
lable = zeros(M,1);

for i=1:M
    if(i<=datamaleRow)
       lable(i,1) = 1;     %���Եı�ǩΪ1��Ů�Եı�ǩΪ0
    elseif(i>datamaleRow)
       lable(i,1) = 0;
    end
end

%��������ְ�
indices=crossvalind('Kfold',data(1:M,N),10);

%������֤k=10��10����������Ϊ���Լ� 
for k=1:10
      test = (indices == k);                            %���test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
      train = ~test;                                    %train��Ԫ�صı��Ϊ��testԪ�صı��
      train_data=data(train,:);                         %�����ݼ��л��ֳ�train����������
 train_lable=lable(train,:);                          %����������Ĳ���Ŀ�꣬�ڱ�������ʵ�ʷ������
      test_data=data(test,:);                           %test������
 test_lable=lable(test,:);

 [Row,Col]=size(test_lable);
 
  t1=cputime;
  models  = fitcnb(test_data,test_lable);  
  predictSex_labels  =  predict(models,train_data);  
  t2=cputime;
  t=t2-t1               %������ѵ������ʱ��
  
  a=0;
  for i=1:Row
    if(predictSex_labels (i,1)==train_lable(i,1))
        a=a+1;
    end
 end
 accuracySexs = a/Row;
  
 end
%�������Ϊ����㷨MLKNN�ļ�����ָ֤�꼰���һ����֤������ͽ������ÿ��ָ�궼��һ��kԪ�ص�������