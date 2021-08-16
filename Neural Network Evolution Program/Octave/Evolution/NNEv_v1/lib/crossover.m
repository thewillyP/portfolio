%Crossover function


function [child1 child2] = crossover(parent1, parent2, parscore1, parscore2)

  [lim j] = min([length(parent1) length(parent2)]);
  sep = abs(length(parent1) - length(parent2));
  
  if j == 2
    a = parscore1/100;
    child1 = [a*parent1(1:lim) parent1(lim+1:end)] + (1-a)*[parent2 zeros(1,sep)];
    child1 = [child1(1:2) parent1(3) round(child1(4:lim)) child1(lim+1:end)];
    
    a = parscore2/100;
    child2 = (1-a)*parent1(1:lim) + a*parent2;
    child2 = [child2(1:2) parent2(3) round(child2(4:lim))];
  else
    a = parscore1/100;
    child1 = a*parent1 + (1-a)*parent2(1:lim);
    child1 = [child1(1:2) parent1(3) round(child1(4:lim))];
    
    a = parscore2/100;
    child2 = (1-a)*[parent1 zeros(1,sep)] + [a*parent2(1:lim) parent2(lim+1:end)];
    child2 = [child2(1:2) parent2(3) round(child2(4:lim)) child2(lim+1:end)];
  endif
  
  if (sum(child1(4:end)) > 30)
    [top j] = max(child1(4:end));
    j = j + 3;
    while sum(child1(4:end)) > 30
      child1(j) -= 1;
      if (child1(j) == 1)
        [top j] = max(child1(4:end));
        j = j + 3;
      endif
    endwhile
  endif
  
  if (sum(child2(4:end)) > 30)
    [top j] = max(child2(4:end));
    j = j + 3;
    while sum(child2(4:end)) > 30
      child2(j) -= 1;
      if (child2(j) == 1)
        [top j] = max(child2(4:end));
        j = j + 3;
      endif
    endwhile
  endif
  
  
end
