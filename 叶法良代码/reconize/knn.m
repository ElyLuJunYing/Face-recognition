function [predictSex_labels, accuracySexs] = knn(dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra)

data = [dataFaceR_maleExtra;dataFaceS_maleExtra;dataFaceR_femaleExtra;dataFaceS_femaleExtra]; %RS混合数据
data = real(data);
[M,N]= size(data);   %数据集为一个M*N的矩阵，其中每一行代表一个样本

datamale = [dataFaceR_maleExtra;dataFaceS_maleExtra];
datafemale = [dataFaceR_femaleExtra;dataFaceS_femaleExtra];
[datamaleRow,datamaleCol] = size(datamale); 
[datafemaleRow,datafemaleCol] = size(datafemale);

%零列
lable = zeros(M,1);

for i=1:M
    if(i<=datamaleRow)
       lable(i,1) = 1;     %男性的标签为1，女性的标签为0
    elseif(i>datamaleRow)
       lable(i,1) = 0;
    end
end

%进行随机分包
indices=crossvalind('Kfold',data(1:M,N),10);

%交叉验证k=10，10个包轮流作为测试集 
for k=1:10
      test = (indices == k);                            %获得test集元素在数据集中对应的单元编号
      train = ~test;                                    %train集元素的编号为非test元素的编号
      train_data=data(train,:);                         %从数据集中划分出train样本的数据
 train_lable=lable(train,:);                          %获得样本集的测试目标，在本例中是实际分类情况
      test_data=data(test,:);                           %test样本集
 test_lable=lable(test,:);

 [Row,Col]=size(test_lable);
 
  t1=cputime;
  models  = fitcnb(test_data,test_lable);  
  predictSex_labels  =  predict(models,train_data);  
  t2=cputime;
  t=t2-t1               %分类器训练所用时间
  
  a=0;
  for i=1:Row
    if(predictSex_labels (i,1)==train_lable(i,1))
        a=a+1;
    end
 end
 accuracySexs = a/Row;
  
 end
%上述结果为输出算法MLKNN的几个验证指标及最后一轮验证的输出和结果矩阵，每个指标都是一个k元素的行向量