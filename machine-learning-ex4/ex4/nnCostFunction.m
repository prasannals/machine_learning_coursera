function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



totalCost = 0;
currentY = zeros(num_labels, 1);

for i = 1:m
	currentX = X(i,:);
	currentX = [1, currentX];
	
	secondLayerOut = sigmoid( Theta1 * currentX' );
	
	secondLayerOut = [1; secondLayerOut];
	
	finalOut = sigmoid(Theta2 * secondLayerOut);
	
	
	correctAns = y(i,:);

	currentY(correctAns) = 1;
	
	
	currentCost = sum( (-currentY .* log(finalOut))  - ((1- currentY) .* log(1- finalOut)) ) ;
	
	totalCost += currentCost;
	
	currentY(correctAns) = 0;

end

J = totalCost * (1/m);


workingTheta1 = Theta1(:, 2:end);

theta1sum = sum(sum(workingTheta1 .^ 2));


workingTheta2 = Theta2(:, 2:end);

theta2sum = sum(sum(workingTheta2 .^ 2));


J = J + ((lambda / (2 * m)) * (theta1sum + theta2sum) );













yVec = zeros(num_labels, 1);

sizT1 = size(Theta1);
sizT2 = size(Theta2);


deltaAccum1 = zeros(sizT1(1), sizT1(2) );

deltaAccum2 = zeros(sizT2(1), sizT2(2) );

for i = 1:m
	% a1 is the input layer activation
	%a1 is 1 x 400
	a1 = X(i,:);
	
	%add the bias unit
	% a1WithBias is 1 x 401
	a1WithBias = [1, a1];
	a1WithBias = a1WithBias';
	%a1WithBias is now 401 x 1

	%perform forward propogation
	%z2 is 25 x 1
	z2 = Theta1 * a1WithBias;

	%a2 is 25 x 1
	a2 = sigmoid(z2);
	%add bias unit
	%a2WithBias is 26 x 1
	a2WithBias = [1; a2];
	
	% z3 is 10 x 1
	z3 = Theta2 * a2WithBias;
	% a3 is 10 x 1
	a3 = sigmoid(z3);

	% correctPred is just one number
	correctPred = y(i);
	% yVec is 10 x 1
	yVec(correctPred) = 1;
	






	%delta3 is 10 x 1
	delta3 = a3 - yVec;

	%delta2 is 25x 1 here
	delta2 = (Theta2(:, 2:end)' * delta3);
	%removing the first column (as it is the bias value)
	
	delta2 = delta2	.* sigmoidGradient(z2);
	
	%fprintf("size of deltaAccum1\n");
	%size(deltaAccum1)

	%fprintf("size of delta2\n");
	%size(delta2)
	
	%fprintf("size of a1\n");
	%size(a1)

	% deltaAccum1 is 25 x 400
	deltaAccum1 = deltaAccum1 + (delta2 * a1WithBias');

	% deltaAccum2 is 10 x 25
	deltaAccum2 = deltaAccum2 + (delta3 * a2WithBias');

	%reset yVec
	yVec(correctPred) = 0;

end


Theta1_grad = (1/m) .* deltaAccum1;
Theta2_grad = (1/m) .* deltaAccum2;


t1FirstCol = Theta1_grad(:, 1);
t2FirstCol = Theta2_grad(:, 1);

Theta1_grad = Theta1_grad + ( (lambda/m) * Theta1) ;
Theta2_grad = Theta2_grad + ( (lambda/m) * Theta2) ;

Theta1_grad(:, 1) = t1FirstCol;
Theta2_grad(:, 1) = t2FirstCol;







% IGNORE THAT FUCKING i COMPLETELY. l is the level of the network
% j is the input (probably. might be the level too. But here's
% the thing - IGNORE THAT FUCKING i. Good. 







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
