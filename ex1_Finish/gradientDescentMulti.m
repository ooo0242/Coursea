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

theta_number = theta;	
	
for i=1:m	
	xi = X(i,:)';	
	yi = y(i,:);
	
	for j=1:size(X,2)
		x = X(i,j);	
		theta_number(j,1) = theta_number(j,1) -(alpha/m)*(sum((theta'*xi)-yi))*x;
		end	
	%x = X(i,:);	
	%theta_number = theta_number-(alpha/m)*(sum((theta'*xi)-yi))*x;
	end
	
theta = theta_number;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
	%fprintf('DEBUG : Value of cost function = %f\n',J_history(iter));

end

end
