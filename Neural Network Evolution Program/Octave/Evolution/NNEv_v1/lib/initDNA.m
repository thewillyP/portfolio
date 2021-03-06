# PART OF RANDOM INITIALIZER

% DNA creates a vector of random values generated by the ranges set. len determines size,
% randFun is a vector of corresponding functions to apply, and rangevec is the corrsponding vector of ranges

function idna = initDNA(len, randFun, rangevec)
    idna = [];
    for i = 1:len
      idna = [idna randMod(randFun{i}, rangevec(:,i))];
    end
end