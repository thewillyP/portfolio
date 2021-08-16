% mutation function
% a is the mutation rate
function M = mutate(thing, a, funcvec, rangevec)
  for i = 1:length(thing)
    if ((1-(1-a)^length(thing)) > rand)
      if (i <= 3)
        thing(i) = randMod(funcvec{i},rangevec(:,i));
        if (i == 3)
          thing = cell2mat(addAllele({thing(1:3)},rangevec(:,4)));
          break
        endif
      else
        v = [thing(4:i-1) thing(i+1:end)];
        thing(i) = randi((30 - sum(v)));
      endif
    endif
  end
  M = thing;
end