function [V, Z] = forwardProp(thetas, X,m)
  
  %Inputs a cell of thetas, the data set, and the input size
  %Returns a cell of activation for each layer from 2 to num_layer. Each matrix in the cell is a m by n matrix. 
  %V is the cell of activ. Z is the cell of pre-activ
  len = size(thetas,1);
  V = cell(len, 1);
  Z = cell(len,1);
  a = X;
  for i = 1:len
    a = [ones(m,1) a];
    Z{i} = a*thetas{i}';
    a = sigmoid(Z{i});
    V{i} = a;
    
  end
 
 
end
