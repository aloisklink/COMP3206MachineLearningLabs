%% Lab 5
%% Using Radial Basis Functions
%% Loading and Normalizing the Data
% The following code loads the housing data from the previous lab,
% normalizes it so that the data has zero mean and unit standard deviation,
% and then splits the data into training and testing datasets.

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
percent = 0.80; % percentage that is training data
selection = randperm(valuesSize);
% creating training data
currentIndexRange = 1:round(valuesSize*percent);
Xtr = Y(selection(currentIndexRange),:);
ytr = f(selection(currentIndexRange));
% creating testing data
currentIndexRange = (round(valuesSize*percent)+1):valuesSize;
Xtst = Y(selection(currentIndexRange),:);
ytst = f(selection(currentIndexRange));

% sets the width of the basis function to a sensible scale.
sig = norm(Xtr(ceil(rand*Ntr),:)-Xtr(ceil(rand*Ntr),:));
basisFunction = @(distance) exp(-norm(distance)/sig^2);

%% Results for K = Ntr/10
K = round(size(Xtr,1)/10);
yh = radialBasis(Xtr, ytr, basisFunction, K, Xtr);
plot(-2:2,-2:2, ytr, yh, 'rx', 'LineWidth', 2), grid on
title('RBF Prediction on Training Data', 'FontSize', 16);
xlabel('Target', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);

