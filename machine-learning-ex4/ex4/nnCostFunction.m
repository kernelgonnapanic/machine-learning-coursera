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

X = [ones(m, 1) X];

J = 0;
for i=1:m
  z_2 = Theta1 * X(i,:)';
  a_2 = [1; sigmoid(z_2)];

  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);


  h = a_3;

  boo=1:num_labels;
  yi = y(i,:)==boo;
  a = -yi'.*log(h);
  b = (ones(num_labels,1)-yi').*log(ones(num_labels,1) - h);
  J = J + 1/m * sum(a - b);
end 
  t1 = Theta1;
  t2 = Theta2;
  [a1, a2] = size(t1);
  [b1, b2] = size(t2);
  t1(:, 1) = zeros(a1, 1);
  t2(:, 1) = zeros(b1, 1);
  a = t1(:);
  b = t2(:);
  reg = lambda / 2 / m * (sum(a.^2) + sum(b.^2));

  J = J + reg;



% -------------------------------------------------------------
%boo=1:num_labels;
%for t=1:1
 % z_2 = Theta1 *  X(t,:)';
 % a_2 = [1; sigmoid(z_2)];

 % z_3 = Theta2 * a_2;
  %a_3 = [sigmoid(z_3)];

 % yi = y(t,:)==boo;
  %err_3 = a_3 - yi';
  %err_2 = (Theta2(:,2:end)'*err_3).*sigmoidGradient(z_2);

  %Theta1_grad = Theta1_grad + err_2 .* X(t,:);
  %Theta2_grad = Theta2_grad  + err_3 .* a_2';
%end
%a1 equals the X input matrix with a column of 1's added (bias units)z2 equals the product of a1 and Θ1a2 is the result of passing z2 
%through g()a2 then has a column of 1st added (bias units)z3 equals the product of a2 and Θ2a3 is the result of passing z3 through g()

a_1 = X;
z_2 = a_1 * Theta1';
a_2 = [ones(size(z_2)(1), 1) sigmoid(z_2)];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

y_matrix = eye(num_labels)(y,:);
err_3 = a_3-y_matrix;
err_2 = err_3 * Theta2(:,2:end) .* (sigmoid(z_2).*(1-sigmoid(z_2)));

Theta1_grad = 1 / m .* (err_2' * a_1);
Theta2_grad = 1 / m .* (err_3' * a_2);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
