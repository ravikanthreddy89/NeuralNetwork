function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of training set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

load('mnist_all.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data=load('mnist_all.mat');


% vectors for output labels
% vector0=[1 0 0 0 0 0 0 0 0 0];
% vector1=[0 1 0 0 0 0 0 0 0 0];
% vector2=[0 0 1 0 0 0 0 0 0 0];
% vector3=[0 0 0 1 0 0 0 0 0 0];
% vector4=[0 0 0 0 1 0 0 0 0 0];
% vector5=[0 0 0 0 0 1 0 0 0 0];
% vector6=[0 0 0 0 0 0 1 0 0 0];
% vector7=[0 0 0 0 0 0 0 1 0 0];
% vector8=[0 0 0 0 0 0 0 0 1 0];
% vector9=[0 0 0 0 0 0 0 0 0 1];

vector0=0;
vector1=1;
vector2=2;
vector3=3;
vector4=4;
vector5=5;
vector6=6;
vector7=7;
vector8=8;
vector9=9;


%pick 1000 examples from each trainging matrix as validation data and rest
%as traning data
validation_data(1:1000,:)=featureNormalize(double(data.train0(1:1000,:)));
validation_data(1001:2000,:)=featureNormalize(double(data.train1(1:1000,:)));
validation_data(2001:3000,:)=featureNormalize(double(data.train2(1:1000,:)));
validation_data(3001:4000,:)=featureNormalize(double(data.train3(1:1000,:)));
validation_data(4001:5000,:)=featureNormalize(double(data.train4(1:1000,:)));
validation_data(5001:6000,:)=featureNormalize(double(data.train5(1:1000,:)));
validation_data(6001:7000,:)=featureNormalize(double(data.train6(1:1000,:)));
validation_data(7001:8000,:)=featureNormalize(double(data.train7(1:1000,:)));
validation_data(8001:9000,:)=featureNormalize(double(data.train8(1:1000,:)));
validation_data(9001:10000,:)=featureNormalize(double(data.train9(1:1000,:)));

validation_label(1:1000,:)=repmat(vector0(1,:),1000,1);
validation_label(1001:2000,:)=repmat(vector1(1,:),1000,1);
validation_label(2001:3000,:)=repmat(vector2(1,:),1000,1);
validation_label(3001:4000,:)=repmat(vector3(1,:),1000,1);
validation_label(4001:5000,:)=repmat(vector4(1,:),1000,1);
validation_label(5001:6000,:)=repmat(vector5(1,:),1000,1);
validation_label(6001:7000,:)=repmat(vector6(1,:),1000,1);
validation_label(7001:8000,:)=repmat(vector7(1,:),1000,1);
validation_label(8001:9000,:)=repmat(vector8(1,:),1000,1);
validation_label(9001:10000,:)=repmat(vector9(1,:),1000,1);

siz=size(data.train0(1001:end,:),1);
train_data(1:siz,:)=featureNormalize(double(data.train0(1001:end,:)));
train_label(1:siz,:)=repmat(vector0(1,:),siz,1);

old_siz=siz;
siz=size(data.train1(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train1(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector1(1,:),siz-old_siz,1);

old_siz=siz;
siz=size(data.train2(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train2(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector2(1,:),siz-old_siz,1);

old_siz=siz;
siz=size(data.train3(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train3(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector3(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.train4(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train4(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector4(1,:),siz-old_siz,1);

old_siz=siz;
siz=size(data.train5(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train5(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector5(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.train6(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train6(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector6(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.train7(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train7(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector7(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.train8(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train8(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector8(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.train9(1001:end,:),1)+old_siz;
train_data(old_siz+1:siz,:)=featureNormalize(double(data.train9(1001:end,:)));
train_label(old_siz+1:siz,:)=repmat(vector9(1,:),siz-old_siz,1);


old_siz=0;
siz=size(data.test0,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test0(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector0(1,:),siz-old_siz,1);

old_siz=siz;
siz=size(data.test1,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test1(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector1(1,:),siz-old_siz,1);

old_siz=siz;
siz=size(data.test2,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test2(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector2(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.test3,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test3(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector3(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.test4,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test4(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector4(1,:),siz-old_siz,1);

old_siz=siz;
siz=size(data.test5,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test5(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector5(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.test6,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test6(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector6(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.test7,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test7(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector7(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.test8,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test8(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector8(1,:),siz-old_siz,1);


old_siz=siz;
siz=size(data.test9,1)+old_siz;
test_data(old_siz+1:siz,:)=featureNormalize(double(data.test9(:,:)));
test_label(old_siz+1:siz,:)=repmat(vector9(1,:),siz-old_siz,1);

end

