function R = randomInitWeights(layer_dimensions);
  %Takes in a vector of the NN architecture
  %Returns a cell of vectors with random weighs corresponding to that architecure
  s = size(layer_dimensions,1)-1;
  R = cell(s,1);
  e = 0; %init_epsilon unit
  
  for i = 1:s
    lin = layer_dimensions(i)+1; %size of the incoming layer
    lout = layer_dimensions(i+1); %size of the outgoing layer
    e = sqrt(6)/sqrt(lin+lout);
    R{i} = rand(lout,lin) *2*e - e;
    
  end
  
  
end
