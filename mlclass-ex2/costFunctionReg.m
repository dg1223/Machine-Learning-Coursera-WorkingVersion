function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis
H = 0;
H = sigmoid(X*theta);	% Study DIMENSION ANAYSIS @ discussion forum

% the cost function
part1 =  (-y).*log(H);
part2 = (1 - y).*log(1 - H);
part3 = 0;
temp = theta;
temp(1) = 0;		% nullifying theta(0) for regularization

%for j = 2:length(theta),
%	part3 = part3 + theta(j)^2;
%end
%part3_final = (0.5*lambda/m) * part3;

part3_final = (0.5*lambda/m) * sum(temp.^2);

J = (1/m) * sum(part1 - part2) + part3_final;	% cost

% gradient calculation
grad(1) = (1/m) * sum((H - y).*(X(:,1)));

for i = 2:size(X,2),
	grad(i) = ((1/m) * sum((H - y).*(X(:,i)))) + (lambda/m)*theta(i);
end

% =============================================================

end
