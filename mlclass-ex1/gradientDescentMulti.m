function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
J = [];

X_norm = featureNormalize(X);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	
	H = 0;
	for i = 1:length(theta),
		H = H + theta(i)*X_norm(:,i);			% compute the value of h(theta)
	end
	
	% compute values of theta
	for j = 1:length(theta),
		theta(j) = theta(j) - alpha * (1/m) * sum((H - y) .* X_norm(:,j));
	end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X_norm, y, theta);

end

end
