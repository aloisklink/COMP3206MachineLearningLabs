%% Lab Four
%% Linear Least Square Regression
% The following code implements Linear Least Square Regression. It tries to
% find a linear line that has the lowest square error. 

% Load Boston Housing Data from UCI ML Repository 
load -ascii housing.data;
% Normalize the data, zero mean, unit standard deviation 
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

% Least squares regression as pseudo inverse
leastSquareReg = @(Ytr, Ftr, Yinput) ...
    Yinput*((Ytr'*Ytr)\Ytr'*Ftr);
fh = leastSquareReg(Y,f,Y);
figure(1), clf,
plot(f, fh, 'r.', 'LineWidth', 2),
grid on
s=getenv('USERNAME');
xlabel('True House Price', 'FontSize', 14)
ylabel('Prediction', 'FontSize', 14)
title(['Linear Regression: ' s], 'FontSize', 14)

%% The Standard Error of Regression (SER)
% The following code is calculate the Standard Error of Regression (SER).
% This is roughly equivalant to the standard deviation of the found data.

squareError = @(Factual, Fpredict) ...
	(Factual - Fpredict).'*(Factual-Fpredict);

SER = @(Ytr, Ftr, Ytst, Ftst) ...
	sqrt(   squareError(Ftst, leastSquareReg(Ytr, Ftr, Ytst) ) ...
		/   (size(Ytr,1) - size(Ytr,2) ) ...
	);

%% K-Fold Cross-Validation
% K-Fold Cross-Validation involves splitting the data in K-sets. Then
% one set is taken for testing and the other sets for training, the next
% set is used for testing and the other for training, and so forth until
% all of the K-sets have been used for testing once, and for training K-1
% times.

% The following code splits the code into K-sets
valuesSize = size(Y,1);
selection = randperm(valuesSize);
k = 10; %how many partitions do we want
dataset(k).in = []; %preallocating

% Splits the data into k (roughly) equal datasets.
for i = 1:k
    currentIndexRange = round(valuesSize*((i-1)/k))+1:round(valuesSize*(i/k));
    dataset(i).in = Y(selection(currentIndexRange),:);
    dataset(i).out = f(selection(currentIndexRange));
end

% RMS is the square-root of the mean of all the squared inputted values.
rms = @(input) sqrt(mean(input.^2));
rmsErrorTr = zeros(k,1);
rmsErrorTest = zeros(k,1);
SERFound = zeros(k,1);

% Tests the data with k-1 datasets for training, and k dataset for testing
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
    
    % See how accurate regression is on training dataset
    Ftrpredict = leastSquareReg(Ytr,Ftr,Ytr);
    rmsErrorTr(i) = rms(Ftrpredict - Ftr);
    
    % See how accurate regression is on testing dataset
    Fpredict = leastSquareReg(Ytr,Ftr,Ytst);
    rmsErrorTest(i) = rms(Fpredict - Ftst);
    
	SERFound(i) = SER(Ytr, Ftr, Ytst, Ftst);
end

disp(['Average Training Root Mean Square Error is ' num2str(mean(rmsErrorTr)) ]);
disp(['Average Test Root Mean Square Error is ' num2str(mean(rmsErrorTest)) ]);
disp(['Average SER is ' num2str(mean(SERFound)) ]);

