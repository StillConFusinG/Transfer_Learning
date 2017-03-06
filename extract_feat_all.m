% Code to extract cNN Features from images and distribute them randomly into 
% training and testing set 

clear all; close all;
rng('default');
net = load('imagenet-vgg-f.mat');
cnnModel.net = vl_simplenn_tidy(net);
cnnModel.net.normalization = cnnModel.net.meta.normalization;
cnnModel.net.classes = cnnModel.net.meta.classes;
save('cnnModel.mat', 'cnnModel');
load('cnnModel.mat');
cnnModel.info.opts.batchSize = 200;
 
imset = dir('../train/*.jpg'); names = {imset.name};
index = randperm(numel(names));tr_size = 5000; 
index_train = index(1:tr_size); 

imageSize = cnnModel.net.normalization.imageSize(1:3);
 
trainingImages = zeros([imageSize tr_size],'single');
trainingLabels = zeros(tr_size,1);
for jj = 1:tr_size
    trainingImages(:,:,:,jj) = imresize(im2single(imread(strcat('../train/',...
                                        names{1,index_train(jj)}))),imageSize(1:2));
    if strfind(names{1,index_train(jj)}, 'cat')
        trainingLabels(jj) = 1;
    else
        trainingLabels(jj) = 2;
    end
end
% % save('trainingImages.mat', 'trainingImages', '-v7.3');
save('trainingLabels.mat', 'trainingLabels');

[~, cnnFeat_train, timeCPU] = cnnPredict(cnnModel,trainingImages,'UseGPU',false);
save('cnnFeat_train.mat', 'cnnFeat_train', '-v7.3');
size(cnnFeat_train)
size(trainingLabels)


test_size=12250; index_test = index(12751:25000);
testingImages = zeros([imageSize test_size],'single');
testingLabels = zeros(test_size,1);
for jj1 = 1:test_size
   testingImages(:,:,:,jj1) = imresize(im2single(imread(strcat('../train/',...
                                       names{1,index_test(jj1)}))),imageSize(1:2));
   if strfind(names{1,index_test(jj1)}, 'cat')
       testingLabels(jj1) = 1;
   else
       testingLabels(jj1) = 2;
   end
end
save('testingImages.mat', 'testingImages', '-v7.3');
save('testingLabels.mat', 'testingLabels');

[~, cnnFeat_test, timeCPU] = cnnPredict(cnnModel,testingImages,'UseGPU',false);
save('cnnFeat_test.mat', 'cnnFeat_test', '-v7.3');