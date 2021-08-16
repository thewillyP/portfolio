function deltas = getDeltas(thetas, Z, hx, y, m)
  %Function that outputs a cell of deltas for each layer. Omits the theta for the bias unit.
  %Takes in the thetas, Z is the theta*X for each layer, the forward prop output, and y. Also m number of training sets
  %We have to flip thetas so that we get it to be in descending order(num_layer to 2) to preform for loop
  %Then flip deltas back to go from 2 to num_layer
  len = size(thetas,1);
  deltas = cell(len,1);
  thetas = flipud(thetas);
  Z = flipud(Z)(2:end,:); %Have to flip Z's because it goes from layer 1 to num_layer. Because its first index now is the pre-activation for...
  %... the num_layer, that's useless so we omit it
  
  %First, get the first delta. 
  deltas(1) = hx - y;
  %Next, get all deltas down to layer 2

  
  for i =  1:len-1
    deltas{i+1} = deltas{i} * thetas{i}(:,2:end) .* [sigmoidGradient(Z{i})];        
  end
  deltas = flipud(deltas); %Flip deltas to go from layer 2 to num_layer
  
end
