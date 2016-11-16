%% Lab Four
%% Linear Least Square Regression
% The following code implements Linear Least Square Regression. It tries to
% find a linear line that has the lowest square error (ie the error (y-distance)
% between the actual point and the regression line, squared).

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
w = ((Y'*Y)\Y'*f);
fh = leastSquareReg(Y,f,Y);
figure(1), clf,
plot(f, fh, 'r.', 'LineWidth', 2),
grid on
s='ak9g14';
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

%% Regression using the CVX Tool
% This finds the line that has the smallest distance between the regression
% line and the actual points.
cvx_begin quiet
variable w1( p+1 );
minimize norm( Y*w1 - f )
cvx_end
fh1 = Y*w1;

%% Sparse Regression
% Sparse Regression is normally used when the number of training data is
% lower than the dimensionality of the data. Because of this, sparse
% regression is used to reduce the dimensionality of the data, so that
% linear regressions works, even though the number of training data is
% small.

% The below code is similar to a normal regression function, except the
% error term now includes a penalty for having large weights.
gammas = linspace(0.01, 40, 100);
iNzero = zeros(size(gammas,2), 1);
for i=1:size(gammas,2)
	cvx_begin quiet
	variable w2( p+1 );
	minimize( norm(Y*w2-f) + gammas(i)*norm(w2,1) );
	cvx_end
	relevantVariables = find(abs(w2) > 1e-6);
	if i == 20
		fh2 = Y*w2;
		plot(f, fh, '.', f, fh1, 'co', f, fh2, 'mx','LineWidth', 2),
		title(['Regression vs Sparse Regression for Gamma = ' num2str(gammas(i))]);
		legend('Least Square Regression','CVX Regression', 'Sparse Regression');
		xlabel('Original Values'); ylabel('Values found using Regression');
		disp(['Relevant variables for gamma = ' num2str(gammas(i))]);
		disp(relevantVariables);
	end
	iNzero(i) = size(relevantVariables,1);
end
figure(2)
plot(gammas, iNzero, 'LineWidth', 2),
title(['Significant Variables in Sparse Regression']);
xlabel('Penalty Complexity Weight: \gamma'); ylabel('Significant Variables (Weights above 1e-6)');
figure(1)
