function T = testAcc(X, y, thetas)
  %Tests the accuracy of the NN
  m = size(X,1);
  activ = forwardProp(thetas, X, m);
  hx = activ{size(activ,1)};
  [dummy predict] = max(hx, [], 2);
  T = mean(double(predict==y)) * 100;  
  
end
