%% Lab 5
%% Using Radial Basis Functions
%% Loading and Normalizing the Data
% The following code loads the housing data from the previous lab,
% normalizes it so that the data has zero mean and unit standard deviation,
% and then splits the data into training and testing datasets.
clear all
% Loads the housing data from Lab4
load -ascii '../Lab4/housing.data';
% Normalize the data to zero mean and unit standard deviation 
[N, p1] = size(housing);
p = p1-1;
Y = [housing(:,1:p) ones(N,1)];
for j=1:p 
    Y(:,j)=Y(:,j)-mean(Y(:,j));
    Y(:,j)=Y(:,j)/std(Y(:,j));
end
f = housing(:,p1);
f = f - mean(f);
f = f/std(f);

valuesSize = size(Y,1);
percent = 0.65; % percentage that is training data
selection = randperm(valuesSize);
% creating training data
currentIndexRange = 1:round(valuesSize*percent);
Xtr = Y(selection(currentIndexRange),:);
ytr = f(selection(currentIndexRange));
% creating testing data
currentIndexRange = (round(valuesSize*percent)+1):valuesSize;
Xtst = Y(selection(currentIndexRange),:);
ytst = f(selection(currentIndexRange));

Ntr = size(Xtr,1);
% sets the width of the basis function to a sensible scale.
sig = norm(Xtr(ceil(rand*Ntr),:)-Xtr(ceil(rand*Ntr),:));
basisFunction = @(distance) exp(-norm(distance)/sig^2);

%% The Radial Basis Function Model
%   A radial basis function model uses the sum of many weighted radial
%   basis functions to approximate a function.
% TODO: INCLUDE CODE HERE

%% Results for K = Ntr/10

K = round(size(Xtr,1)/10);
ytrpredict = radialBasis(Xtr, ytr, basisFunction, K, Xtr);
ytstpredict = radialBasis(Xtr, ytr, basisFunction, K, Xtst);
plot(-2:2,-2:2, 'DisplayName', 'Zero-Error Line');
hold on; axis([-2 3.1 -2.5 2])
plot(ytr, ytrpredict, 'b.', 'LineWidth', 2, 'DisplayName', 'Training Data');
plot(ytst, ytstpredict, 'rx', 'LineWidth', 1, 'DisplayName', 'Test Data'), grid on
title('RBF Prediction on Data', 'FontSize', 16);
xlabel('Target', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);
legend('Location', 'Southeast');
hold off;
%% RMS Error for a Range of Ks
% The following code plots the root-mean-square (RMS) error on the RBF model's
% output, when different K's are used. As can be seen, it appears that
% the test data does pretty well, until K reaches around 20,
% and the model overfits towards the training data, becoming too accurate.

rms = @(input) sqrt(mean(input.^2));
for K = 1:40
	ytrpredict = radialBasis(Xtr, ytr, basisFunction, K, Xtr);
	trError(K) = rms(ytrpredict-ytr);
	ytstpredict = radialBasis(Xtr, ytr, basisFunction, K, Xtst);
	tstError(K) = rms(ytstpredict-ytst);
end
K = 1:40;
plot(K, trError, 'LineWidth', 2, 'DisplayName', 'Training RMS Error'); hold on;
plot(K, tstError, 'LineWidth', 2, 'DisplayName', 'Test RMS Error'), grid on
title('RMS Error on Data with differing K', 'FontSize', 16);
xlabel('K', 'FontSize', 14);
ylabel('RMS Error', 'FontSize', 14);
legend('Location', 'Northeast');
hold off;

%% Comparing Linear Least Square Regression to RBF on Housing Data
% Least squares regression as pseudo inverse
leastSquareReg = @(Ytr, Ftr, Yinput) ...
    Yinput*((Ytr'*Ytr)\Ytr'*Ftr);

% The following code splits the code into K-sets
valuesSize = size(Y,1);
selection = randperm(valuesSize);
k = 20; %how many partitions do we want
dataset(k).in = []; %preallocating

% Splits the data into k (roughly) equal datasets.
for i = 1:k
    currentIndexRange = round(valuesSize*((i-1)/k))+1:round(valuesSize*(i/k));
    dataset(i).in = Y(selection(currentIndexRange),:);
    dataset(i).out = f(selection(currentIndexRange));
end

for i = 1:k 
	
    Ytr = []; Ftr = [];    
    % Makes the training dataset
    for j = 1:k
        if(i ~= j)
            Ytr = [Ytr; dataset(j).in];
            Ftr = [Ftr; dataset(j).out];
        end
    end
    % Makes the testing dataset
    Ytst = dataset(i).in;
    Ftst = dataset(i).out;
	% see how accurate regression is on RBF
	Ntr = size(Xtr,1);
	sig = norm(Xtr(ceil(rand*Ntr),:)-Xtr(ceil(rand*Ntr),:));
	basisFunction = @(distance) exp(-norm(distance)/sig^2);
	K = round(size(Xtr,1)/10);
	
    Fpredict = radialBasis(Ytr, Ftr, basisFunction, K, Ytst);
	rmsErrorRBF(i) = rms(Fpredict - Ftst);
    % See how accurate regression is on testing dataset
    Fpredict = leastSquareReg(Ytr,Ftr,Ytst);
    rmsErrorLinear(i) = rms(Fpredict - Ftst);
end
boxplot([rmsErrorRBF' rmsErrorLinear'], 'Labels', {'RBF', 'Linear'});
title('RMS Error on Housing Data');
axis([0 3 0 3]);
%% Comparing Linear Least Square Regression to RBF on Power Plant
load -ascii 'CCPP/PowerPlant.csv';
% Normalize the data to zero mean and unit standard deviation 
[N, p1] = size(PowerPlant);
p = p1-1;
Y = [PowerPlant(:,1:p) ones(N,1)];
for j=1:p 
    Y(:,j)=Y(:,j)-mean(Y(:,j));
    Y(:,j)=Y(:,j)/std(Y(:,j));
end
f = PowerPlant(:,p1);
f = f - mean(f);
f = f/std(f);

% Least squares regression as pseudo inverse
leastSquareReg = @(Ytr, Ftr, Yinput) ...
    Yinput*((Ytr'*Ytr)\Ytr'*Ftr);

% The following code splits the code into K-sets
valuesSize = size(Y,1);
selection = randperm(valuesSize);
k = 20; %how many partitions do we want
dataset(k).in = []; %preallocating

% Splits the data into k (roughly) equal datasets.
for i = 1:k
    currentIndexRange = round(valuesSize*((i-1)/k))+1:round(valuesSize*(i/k));
    dataset(i).in = Y(selection(currentIndexRange),:);
    dataset(i).out = f(selection(currentIndexRange));
end

for i = 1:k 
	
    Ytr = []; Ftr = [];    
    % Makes the training dataset
    for j = 1:k
        if(i ~= j)
            Ytr = [Ytr; dataset(j).in];
            Ftr = [Ftr; dataset(j).out];
        end
    end
    % Makes the testing dataset
    Ytst = dataset(i).in;
    Ftst = dataset(i).out;
	% see how accurate regression is on RBF
	Ntr = size(Xtr,1);
	sig = norm(Xtr(ceil(rand*Ntr),:)-Xtr(ceil(rand*Ntr),:));
	basisFunction = @(distance) exp(-norm(distance)/sig^2);
	K = round(size(Xtr,1)/10);
	
    Fpredict = radialBasis(Ytr, Ftr, basisFunction, K, Ytst);
	rmsErrorRBF(i) = rms(Fpredict - Ftst);
    % See how accurate regression is on testing dataset
    Fpredict = leastSquareReg(Ytr,Ftr,Ytst);
    rmsErrorLinear(i) = rms(Fpredict - Ftst);
end
boxplot([rmsErrorRBF' rmsErrorLinear'], 'Labels', {'RBF', 'Linear'});
title('RMS Error on Power Plant Data');
axis([0 3 0 1]);