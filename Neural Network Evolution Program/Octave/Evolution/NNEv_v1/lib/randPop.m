# PART OF RANDOM INITIALIZER

% Initiliazes a random population of "size" amount. 
% Look at initDNA documentation for info on rest of parameters.
function pop = randPop(siz, len, randFun, rangevec)
  op = @(L) initDNA(L, randFun, rangevec(:,1:3));
  pop = arrayfun(op, ones(siz,1)*len, "UniformOutput", false);
  pop = addAllele(pop, rangevec(:,4));
end