function [J, grad] = nnCostFunction(theta_rolled, layer_dimensions, X, y, lambda)
  
  %theta_rolled is a vector of rolled theta_rolled
  %layer_dimensions is a vector with the size of each layer, index corresponds to layer. Index 1 are the input size
  #X is the data 
  #y is the training set
  %lambda for regularization
  
  %Function computes the neural network cost function for a generic neural network of an arbitrary number of sizes and layers.
  %Returns the cost, the gradient
    

  %Useful Vars
  num_layer = size(layer_dimensions,1);
  m = size(X,1);
  num_labels = layer_dimensions(num_layer);
  
  %Unroll vectors for any arbitrary number of layers. thetas is a cell
  thetas = unroll(theta_rolled, layer_dimensions); %rolled theta, the vector of layer sizes, the size of that vector
  
  
  %Forward Propagation
  [activ, Z] = forwardProp(thetas, X, m); % activ is a cell of activation from 2 to num_layer. Z is a similar cell of the pre-activation numbers
  %get hx
  hx = activ{num_layer-1};

  %cost function
  J = (sumDiag(y'*log(hx) + (1-y)'*log(1-hx)))/(-m) + squareAllTheta(thetas, lambda, m);
  
  %Back Propagation
  deltas = getDeltas(thetas, Z, hx, y, m); %Gets a cell of the deltas in ascending order, from layer 2 to num_layer
  D = getDeriv(deltas, [X; activ], thetas, lambda, layer_dimensions, m); #Finally gets a cell of the derivatives
  
  grad = roll(D); #Rolls cell D into a vector. 

  
end
