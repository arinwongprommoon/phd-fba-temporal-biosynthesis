cd /home/arin/git/GECKO/geckomat
model = importModel('yeast-GEM-8_6_2.xml')
modelname = 'ecYeast8';
[ecModel, ecModel_batch] = enhanceGEM(model,'COBRA', modelname, '8_6_2');
cd ../models
save([modelname '/' modelname '.mat']','ecModel');
save([modelname '/' modelname '_batch.mat'], 'ecModel_batch');
