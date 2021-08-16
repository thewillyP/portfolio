function R = squareAllTheta(thetas, lambda, m)
  %Computes the regularization for the cost function for all layers of theta
  %Takes in the cell of thetas, lambda, and m the size of training set
  len = size(thetas,1);
  R = 0;
  for i = 1:len
    theta_mod = thetas{i}(:,2:end);
    R = R + (lambda/(2*m))*sumDiag(theta_mod*theta_mod');
  end
  
end
