function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma = [0.01;0.03;0.1;0.3;1;3;10;30];
[C,sigma] = meshgrid(C, sigma);
C_sigma_pairs = [C(:), sigma(:)];  

prediction = zeros(size(C_sigma_pairs,1), size(yval, 1));
prediction_error = zeros(size(prediction,1), 1);
for i = 1:size(prediction,1)
  C = C_sigma_pairs(i, 1); sigma = C_sigma_pairs(i, 2);
  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  prediction(i, :) = svmPredict(model, Xval)';
  prediction_error(i) = mean(double(prediction(i,:)' ~= yval));
  fprintf(['Prediction_error: %f\tC: %f\tSigma: %f\t'], prediction_error(i), C, sigma);
end
 [~, lowest_error_index] =  min(prediction_error);
 C = C_sigma_pairs(lowest_error_index, 1);
 sigma = C_sigma_pairs(lowest_error_index, 2);





% =========================================================================

end
