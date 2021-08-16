function score = alphClass(X, y, NN_Arch, lambda, iter)
  
  if (NN_Arch(end) >= 3)
    T = y == 1:(NN_Arch(end));
  else
    T = y == 1;
  endif
  
  training_set = X(1:16000,:);
  training_label = T(1:16000,:);
    
  test_set = X(16001:20000,:);
  test_label = y(16001:20000,:);
  
  %Trains network
  [nnParams cost] = testNN(NN_Arch, training_set, training_label, lambda, iter);
  score = testAcc(test_set, test_label,nnParams);
    
end