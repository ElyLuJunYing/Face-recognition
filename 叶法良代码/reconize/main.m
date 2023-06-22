clc;
load dataFaceS_male
load dataFaceS_female
load dataFaceR_male
load dataFaceR_female
load ev.mat 
load train_list
dimension=100;

%ÑµÁ·
[dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra]= sexDistanceJ2(dataFaceR_male,dataFaceR_female,dataFaceS_male,dataFaceS_female,dimension);
[predictSex_label, accuracySex] = sex(dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra);
[predictSex_labels, accuracySexs] = knn(dataFaceR_maleExtra,dataFaceR_femaleExtra,dataFaceS_maleExtra,dataFaceS_femaleExtra)

