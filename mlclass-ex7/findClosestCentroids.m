function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

distance = 0;
min_distance = 0;

% X and centroids are matrices which hold values of datapoints and avg of corresponding
% datapoints respectively. These are not coordinates! only solution: Matrix multiplication
for i = 1:size(X,1),
	%i
	for j = 1:size(centroids,1),
		%j
		if j == 1			% both distances are equal on the 1st iteration
			distance = (X(i,:) - centroids(j,:)) * (X(i,:) - centroids(j,:))';
			min_distance = distance;
			idx(i) = j;		% store centroid index
		else				% compare from 2nd iteration
			distance = (X(i,:) - centroids(j,:)) * (X(i,:) - centroids(j,:))';
			if distance < min_distance;
				min_distance = distance;
				idx(i) = j;
			end			
		end
	end
end




% =============================================================

end

