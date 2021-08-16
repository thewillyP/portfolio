

function allele = addAllele(pop, rangevec)
  LIM = rangevec(2) + 1;
  for i = 1:length(pop)
    lay = pop{i}(3);
    a = zeros(1,lay);
    order = randperm(lay);
    diff = LIM - lay;
    for j = order
      if (diff <= 0)
        a(j) = 1;
      else
        a(j) = randi([1 diff]);
        diff -= a(j);
      endif
    end
    pop{i} = [pop{i} a];
  end
  allele = pop;
end
