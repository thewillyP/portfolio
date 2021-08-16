function G = sigmoidGradient(z)
  %Calculates the sigmoid Gradient
  G = sigmoid(z) .* (1-sigmoid(z));
  
  
end
