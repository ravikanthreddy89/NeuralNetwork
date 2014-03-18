function g = sigmoid(z)
% sigmoid computes sigmoid functoon
% Notice that z can be a scalar, a vector or a matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% @author : Ravikanth Reddy Gudipati
%we have two ways to calculate sigmoid function
%1) manually
%2) using matlab's sigmoid membership function
%advantage of using matlab's sigmoid membership function
%is takes care of z, vector or scalar or matrix.

%g=sigmf(z,[1 0]); 
g=1.0./(1.0+exp(-z));
end
