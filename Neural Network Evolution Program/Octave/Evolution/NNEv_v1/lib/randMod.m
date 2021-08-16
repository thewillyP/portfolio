# PART OF RANDOM INITIALIZER

% Takes in two vecotrs. func is a designated operator/function. vec is a 1 by 2 vector of the range
% where vec(1) is the MIN and vec(2) is the MAX. randMod operates the function on the vector to make
% a random number between those two intervals
function prob = randMod(func, vec)
  name = func2str(func);
  if (strcmp(name, "rand"))
    prob = func(1)*(vec(2)-vec(1)) + vec(1);
  elseif (strcmp(name, "randi"))
    prob = func([vec(1) vec(2)]);
  else
    prob = NaN;
  endif

  
end

