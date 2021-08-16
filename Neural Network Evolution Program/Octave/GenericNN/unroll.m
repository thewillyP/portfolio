function thetas = unroll(theta_rolled, layer_dimensions)
  %theta_rolled is a vector of rolled theta_rolled
  %layer_dimensions is a vector with the size of each layer, index corresponds to layer. Index 1 are the input size
  #X is the data 
  #y is the training set
  %lambda for regularization
  
  num_layer = length(layer_dimensions);
  thetas = cell(num_layer-1,1);
  start = 1;
  stop = 0;
  for i = 1:(num_layer-1)
    stop = stop+(layer_dimensions(i)+1)*layer_dimensions(i+1);
    thetas{i} = reshape(theta_rolled(start:stop), ...
                         layer_dimensions(i+1), layer_dimensions(i)+1);
    start = stop + 1;
  
  end
  
  
end