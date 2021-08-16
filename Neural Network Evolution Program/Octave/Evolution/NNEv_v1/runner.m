%runner for Ev simulator
clear;
%Sets up the environment



addpath('../../Alphabet/Alphabet_Dataset');
addpath('../../GenericNN');
addpath('./lib');

load("letters.mat");

% 16 inputs. 26 outputs;

% -------------------------------------------------------------------
% Sets up the variables for training

  X = training.data;                  %Gets X
  X = featureNormalize(X);            %Feature Normalize
  y = cell2mat(training.textdata);    %Gets y
  y = double(y) - 64;
  
  
% --------------------------------------------------------------------

%Now set up evolution environment
%100 population size
% 3 Necessary genomes. The trailing genes are the number of neurons for each layer
% [lamba iter #hiddenlayers numberofneuronsforeachlayer]
% Ex. [1 50 3 5 5 5]

%Define ranges for each genome
rndm = @rand;
rndmi = @randi;
funcvec = {rndm rndmi rndmi};
rangevec = [[.0001;1000] [1;150] [1;30] [1;30]];

% Define Evolution parameters
gen_i = 2; % # of Generational cycle
pop_size = 15;
keep = round(pop_size*.1); %Number of population who survives after culling
start = 1;
a = .1; %mutation rate

% Creates a cell of _NN species
pop = randPop(pop_size, 3, funcvec, rangevec);

%Seed stores all information across generations
%seed = {};
%load([[pwd "\\Generations\\Gen"] num2str(27) ".mat"]);
%pop = seed(:,1);

% Start Generational cycle
for i = start:gen_i
  
  % Training Neural Networks
  score = zeros(size(pop,1),1);
  for j = 1:size(pop,1)
    len = size(pop{j},2);
    score(j) = alphClass(X, y, [16 ; (pop{j}(:,4:len))' ; 26], pop{j}(1), pop{j}(2));
  end
  [score orig_i] = sort(score, "descend");
  
  seed = pop(orig_i);
  seed(:,2) = num2cell(score);
  filename = [[pwd "\\Generations\\Gen"] num2str(i) ".mat"];
  save (filename, "seed");
  
  %Crossover and mutation
  pop = repopulate(pop(orig_i), score, keep, a, funcvec, rangevec);

end

%Evaluate final performance
score = zeros(size(pop,1),1);
for j = 1:size(pop,1)
  len = size(pop{j},2);
  score(j) = alphClass(X, y, [16 ; (pop{j}(:,4:len))' ; 26], pop{j}(1), pop{j}(2));
end
[score orig_i] = sort(score, "descend");

disp("Final population is");
disp(pop(orig_i));
disp("Final score is");
disp(score);

