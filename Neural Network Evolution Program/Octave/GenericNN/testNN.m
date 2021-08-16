function [nnParams, cost] = testNN(layer_dimensions, X, y, lambda, iter)

  %sets the options for fmincg gradient
  options = optimset("MaxIter", iter);
  
% ----------------------------------------------------------------------------------------
  %Creates the random matrices of theta and rolls them up
  theta_rolled = roll(randomInitWeights(layer_dimensions));
  
% ----------------------------------------------------------------------------------------
  %Temporarily sets up no regulzarization to test the numerical gradient
  temp_lambda = lambda;
  lambda = 0;
  
% ----------------------------------------------------------------------------------------
  %Create shorthand for cost functipn
  costFunc = @(p) nnCostFunction(p, layer_dimensions, X, y, lambda);

% ----------------------------------------------------------------------------------------
  %Training Neural network
  
  lambda = temp_lambda;
  [nnParams, cost] = fmincg(costFunc,theta_rolled,options);
  nnParams = unroll(nnParams,layer_dimensions);
  %disp("New Weights are: ");
  %disp(nnParams);
  
  
end
