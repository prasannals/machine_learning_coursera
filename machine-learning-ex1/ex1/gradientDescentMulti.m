function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



    % X ALREADY HAS THE COEFFICIENT 1 IN THE FIRST COLUMN
	
    % with the current values of theta, compute the predictions
    % for each row of data
    ourPred = (X * theta);
    
    %ourPred
    % ourPred now has predictions for all m rows. find the
    % difference between this and the actual y values
    
    difference = ourPred - y;

    %difference

    % for row i = 1; i <= m; difference(i) represents the
    % difference between the prediction and the actual y
    % for the data set(i.e. row) i


    % for row i = 1; i <= m; difference(i) will be 
    % multiplied to column X(:, 1). this is what is happening
    % here
    rowSums = difference' * X;
	
    %rowSums    

    % rowSums is a 1 * n matrix now. Multiply each of those by
    % 1/m
    rowSums = rowSums .* (1/m);

    % finally multiply each element with alpha
    rowSums = rowSums .* alpha;

    %fprintf("After everything\n");
    %rowSums
    % update the theta. since rowSums is 1 * n, invert it
    theta = theta - rowSums';

    %theta








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
