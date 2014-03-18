function [X_norm] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
%X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%@author : Sushikundu
% mu=mean(X);
% sigma=std(X,1);

% mu_rep=repmat(mu,size(X,1),1);
% sigma_rep=repmat(sigma, size(X,1),1);

% mean_diff=X-mu_rep;
% sigma_div=mean_diff./sigma_rep;

% X_norm=sigma_div;   

a = min (X(:));
b = max (X(:));
ra = 0.9;
rb = 0.1;

X_norm = (((ra - rb) * (X - a))/ (b -a)) + rb;

% ============================================================

end
