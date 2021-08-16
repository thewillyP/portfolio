function R = roll(unrolled)
  %rolls a cell of matrices into a vector. Returns that cell and a matrix where the rows are the dimensions for each matrix
  %R is the vector of rolled. V is the vector of sizes.
  
  R = [];
  for i = 1:size(unrolled,1)
    M = unrolled{i}; %get the matrix
    R = [R; M(:)];
    
  end
  
  
end
