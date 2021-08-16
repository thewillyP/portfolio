
% Selection algorithm
% a is the mutation rate
function newPop = repopulate(pop, score, keep, a, funcvec, rangevec)
  initpool = size(pop,1);
  half_pool_perm = ceil(initpool/2);
  half_pool_fix = fix(initpool/2);
  % Change +1 for odd number values
  lowPreformCull = (randperm((half_pool_perm)) + half_pool_fix)(1:half_pool_perm-keep); %Chooses underpreforming to cull
  highPreformCull = randperm(half_pool_fix)(1:keep);
  cull = [highPreformCull lowPreformCull];
  pop(cull) = [];
  score(cull) = [];
  
  % Crossover and Mutation
  pool = randperm(size(pop,1));
  if (mod(length(pool),2) == 1)
    i = pool(:,end);
    pool = pool(:,1:end-1);
    pop{i+half_pool_perm} = mutate(pop{i},a,funcvec,rangevec);
  endif
  
  if (mod(initpool,2) == 1)
    pop{half_pool_perm} = cell2mat(randPop(1,3,funcvec,rangevec));
  endif
  
  x = 1;
  while x <= length(pool)
    i = pool(x);
    j = pool(x+1);
    parent1 = pop{i};
    parent2 = pop{j};
    [child1 child2] = crossover(parent1, parent2, score(i), score(j));
    pop{i+half_pool_perm} = mutate(child1,a,funcvec,rangevec);
    pop{j+half_pool_perm} = mutate(child2,a,funcvec,rangevec);
    x += 2;
  endwhile
  newPop = pop;    
  
end
