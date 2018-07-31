function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %


fprintf('DEBUG : Value of theta zero = %f\n',theta(1,1));
fprintf('DEBUG : Value of theta one = %f\n',theta(2,1));

theta_zero = theta(1,1);
theta_one = theta(2,1);

for i=1:m
	xi = X(i,:)';
	yi = y(i,:);
	x_zero = X(i,1);
	x_one = X(i,2);
	theta_zero = theta_zero - (alpha/m)*(sum(theta'*xi-yi)*x_zero);
	theta_one = theta_one - (alpha/m)*(sum(theta'*xi-yi)*x_one);
	end
	
theta = [theta_zero; theta_one];
	



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	fprintf('DEBUG : Value of cost function = %f\n',J_history(iter));

end

end
