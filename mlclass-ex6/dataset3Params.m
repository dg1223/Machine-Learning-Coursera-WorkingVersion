function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

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

% initializing variables
len = length(C_vect);
prediction = 0;
cur_err = 0;
prev_err = zeros(size(C_vect,1)^2);
min_err = zeros(3,1);
count = 0;

for i = 1:len,
	for j = 1:len,
		count = count + 1;
		model= svmTrain(X, y, C_vect(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(j)));
		prediction = svmPredict(model, Xval);	% fit model to CV set
		cur_err = mean(double(prediction ~= yval));
		prev_err(count) = cur_err;
		if (prev_err(count) == min(prev_err(1:count))),
			min_err = [C_vect(i); sigma_vect(j); cur_err];
			C = min_err(1);
			sigma = min_err(2);
		end

	end
end



% =========================================================================

end
