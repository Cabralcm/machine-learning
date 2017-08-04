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

% init layer size def
s_1 = input_layer_size;
s_2 = hidden_layer_size;
s_3 = num_labels;
% vectorize classes
y_matrix = bsxfun(@eq, y, 1:num_labels); % m x s_3
bias_unit = ones(m,1);

% forward prop
a_1 = [bias_unit, X]; % m x (s_1 + 1)

z_2 = a_1 * Theta1'; % m x s_2
a_2 = [bias_unit, sigmoid(z_2)]; % m x (s_2 +1)

z_3 = a_2 * Theta2'; % m x s_3
a_3 = sigmoid(z_3); % m x s_3

h = a_3; % m x s_3

% Cost func
J = zeros(m, 1);
for i = 1:m,
    J(i) = (1/m) * ( -y_matrix(i,:) * log(h(i,:)') - (1-y_matrix(i,:)) * log(1-h(i,:)') );
end
J = sum(J);

% Cost reg
Theta1_unbiased = Theta1(:, 2:end); Theta2_unbiased = Theta2(:, 2:end);
Theta1_unbiased_sqr = Theta1_unbiased.^2; Theta2_unbiased_sqr = Theta2_unbiased.^2;
J_reg = (lambda/(2*m)) * ( sum(Theta1_unbiased_sqr(:)) + sum(Theta2_unbiased_sqr(:)) );

J = J + J_reg;

% Back Prop
d_3 = zeros(m, num_labels);
d_2 = zeros(m, hidden_layer_size);
D_2 = zeros(s_3,s_2+1);
D_1 = zeros(s_2,s_1+1);
for i = 1:m,
    d_3(i,:) = a_3(i,:) - y_matrix(i,:); % 1 x s_3
    d_2(i,:) = d_3(i,:) * Theta2_unbiased .* sigmoidGradient(z_2(i,:)); % 1 x (s_2)
    
    D_2 = D_2 + d_3(i,:)' * a_2(i,:); % s_3 x (s_2+1)
    D_1 = D_1 + d_2(i,:)' * a_1(i,:); % s_2 x (s_1+1)
end
% Gradient
Theta1_reg = (lambda/m) * [zeros(s_2,1), Theta1_unbiased];
Theta2_reg = (lambda/m) * [zeros(s_3,1), Theta2_unbiased];

Theta1_grad = (1/m) * D_1 + Theta1_reg;
Theta2_grad = (1/m) * D_2 + Theta2_reg;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
