function D = getDeriv(deltas, activ, thetas, lambda, layer_dimensions, m)
  %Outputs a cell of derivatives for each layer for the gradient. 
  %deltas is a cell of the deltas in the NN. activ is a cell of the activation outputs for each layer (with the X as the first layer).
  %thetas is a cell of the theta for each layer. lambda is the lambda. layer_dimensions is ths vector of layer sizes. m is the size of training sets
  
  %Adds regularization 
  
  D = cell(size(thetas,1),1);
  for i = 1:size(thetas,1)
    D{i} = (deltas{i}' * [ones(m,1) activ{i}] + [zeros(layer_dimensions(i+1), 1) thetas{i}(:,2:end)*lambda])/m;
  end
  
end
