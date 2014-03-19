function label = nnPredict(w1, w2, data)
% nnPredict predicts the label of data given the parameter w1, w2 of Neural
% Network.

% Input:
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image
       
% Output: 
% label: a column vector of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%you have w1, w2 and data and return the label
%trainingSize=size(data,1);
label=zeros(size(data,1),1);
for i=1:size(data,1)
    a1=[1 data(i,:)].';
    z1=w1*a1;
    a2=[1 ; sigmoid(z1)];
    z2=w2*a2;
    a3=sigmoid(z2);
    %thisLabel=zeros(10,1);
    pIndex=find(a3>=0.5);
    %thisLabel(pIndex)=1;
    %label(i,pIndex.')=1;
    label(i,1)=sum(pIndex-1);
end

end
