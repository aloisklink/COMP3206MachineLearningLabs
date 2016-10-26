%% Machine Learning Lab One COMP3206
% This Lab covers generating 2-D normal (Gaussian) distributions.
%
% The lab notes can be found here: 
% <https://secure.ecs.soton.ac.uk/notes/comp6229/ml_lab_one2016.pdf>
%% Step 2
% The following shows a histogram of N=1000 uniform random numbers.
N = 1000; x = rand(N,1);
bar(hist(x,10)); %one-liner to save space since two $$/Latex$$ pages is no space
xlabel('Number of Category'); ylabel('Size of Category'); title('Uniform Random Numbers Histogram');
%% 
% The following shows a histogram of N=1000 normal distrubuted random
% numbers.
x = randn(N, 1);
bar(hist(x,10));
xlabel('Number of Category'); ylabel('Size of Category'); title('Normal Distributed Numbers Histogram');
%%
% The following shows a histogram of the sum of 12 uniform distributed
% random numbers minus the sum of 12 other uniform distributed random
% numbers.
%
% This shows a Gaussian distribution, even though only uniform distributed
% values were input. This shows the 
% <https://en.wikipedia.org/wiki/Central_limit_theorem Central Limit Theorem> 
% in action.

x1 = zeros(N,1);
for n=1:N
	x1(n,1) = sum(rand(12,1))-sum(rand(12,1));
end
hist(x1,40);
xlabel('Category Value'); ylabel('Size of Category'); title('Normal Distributed Numbers Histogram');
%% Step 3
% The following factorizes *C* into *A*, which can be confirmed by looking at
% *Ctest*.
C = [2 1 ; 1 2]
A = chol(C); Ctest = transpose(A) * A
%%
% The graph shows that *X* is isotropically Normally distributed, while *Y* is distributed
% roughly correlated to a line. 
X = randn(N,2);
Y = X * A;
plot(X(:,1),X(:,2),'c.', Y(:,1),Y(:,2),'mx');
xlabel('X_1 features'); ylabel('X_2 features'); title('Scatter Plot of X and Y');
%%
% This graph shows *Y* projected onto a line *W* with direction $\theta$. It then
% plots the variance of this for each $\theta$. This makes a sine wave, whose
% maxima and minima correspond to *C*'s eigenvalues, and the turning points
% correspond to the location of *C*'s eigenvectors.
%
% Using the other covariance matrix, *C2*, the sine wave has a $\pi$ added 
% phase.

Nvals = 50;
plotArray = zeros(Nvals,1);
yp = zeros(N, 1, Nvals);
thetaRange = linspace(0,2*pi,Nvals);
for n=1:Nvals
    theta = thetaRange(n);
    W = [sin(theta); cos(theta)];
    yp(:,:, n) = Y * W;
    plotArray(n) = var( yp(:,:, n) );
end
plot(thetaRange, plotArray); ylim([0.8, 3.2]);
xlabel('Angle of Projection Line'); ylabel('Variance'); title('Variance of Y when Projected');
set(gca,'xtick',0:pi/2:2*pi); set(gca,'xticklabel',{'0','\pi/2','\pi','3 \pi/2','2 \pi'});
hold on;

%
% Using Y2, which is calculated from C2, this graph shows Y2 projected onto
% a line W with direction theta. It then
% plots the variance of this for each theta.
%
C2 = [2 -1; -1 2];
A2 = chol(C2);
Y2 = X * A2;
plotArray2 = zeros(Nvals,1);
yp2 = zeros(N, 1, Nvals);
for n=1:Nvals
    theta = thetaRange(n);
    W = [sin(theta); cos(theta)];
    yp2(:,:, n) = Y2 * W;
    plotArray2(n) = var( yp2(:,:, n) );
end
plot(thetaRange, plotArray2, '-');

% Find the Eigenvectors and Eigenvals and plot them
[eigVecs, eigVals] = eig(C);
eigVecAngs(1) = atan(eigVecs(1,1)/eigVecs(2,1)) + 2*pi;
eigVecAngs(2) = atan(eigVecs(1,2)/eigVecs(2,2));
plot([0 2*pi], [eigVals(1) eigVals(1)]); plot([0 2*pi], [eigVals(4) eigVals(4)]);
plot(eigVecAngs(1)*[1 1], [1 3]); plot(eigVecAngs(2)*[1 1], [1 3]);
hold off;

