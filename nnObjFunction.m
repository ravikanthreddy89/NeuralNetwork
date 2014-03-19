function [obj_val, obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, training_data,...
                                    training_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.
w1 = reshape(params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%n_input is your input size
%n_hidden is your number of hidden units
%n_class is you output units

Jtheta=0;
cost=0;
bigDelta1=zeros(size(w1));
bigDelta2=zeros(size(w2));
%perform forward propogation fist
N=size(training_data,1);
for i=1:N
    a1=[1 training_data(i,:)].';
    z1=w1*a1;
    a2=[1 ; sigmoid(z1)];
    z2=w2*a2;
    a3=sigmoid(z2);
    %here ends forward propogation
    
    %compute cost for this example
    hTheta=a3;
    %dude : you are using the integer value label here too
    %positiveCost=(training_label(i,:).').*log(hTheta);
    %negativeCost=((1-training_label(i,:)).').*log(1-hTheta);

    output=zeros(10,1);
    output(training_label(i)+1)=1;
        
    positiveCost=output.*log(hTheta);
    negativeCost=((1-output)).*log(1-hTheta);
    cost=cost+sum(positiveCost+negativeCost);
    
    %start backward propogation with deltas
    %dude : here is the error. training_label is not 1 of k encoded do that
    %first
    delta3=a3-output;
    %delta3=a3-training_label(i,:).';
    delta2=(w2')*delta3.*(a2.*(1-a2));
    
    bigDelta2=bigDelta2+(delta3*(a2.'));
    bigDelta1=bigDelta1+(delta2(2:end,:)*(a1.'));
end
m=size(training_data,1);
w1_nonbias=[ zeros(size(w1,1),1) lambda*w1(:,2:end)];
w2_nonbias=[ zeros(size(w2,1),1) lambda*w2(:,2:end)];

grad_w1=(bigDelta1/m)+w1_nonbias;
grad_w2=(bigDelta2/m)+w2_nonbias;

obj_val=(cost/(-m))+(((sumsqr(w1(:,2:end))+sumsqr(w2(:,2:end)))*lambda)/(2*m));

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 and grad_w2
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];

end
