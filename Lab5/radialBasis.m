function [ Fout ] = radialBasis( Xtr, Ftr, basisFunction, K, Xin )
%radialBasis Predicts answers using a radial basis function model
%   A radial basis function model uses the sum of many weighted radial
%   basis functions to approximate a function. BasisFunction is a
%	function of distance, ie BasisFunction(distance).
	Ntr = size(Xtr,1);
	% performs K-Mean clustering to find the centers for the basis function
	[C] = kmeans(Xtr, K);
	A = zeros(Ntr,K);
	% construct the design matrix
	for i=1:Ntr
		for j=1:K
			A(i,j) = basisFunction(Xtr(i,:) - C(j,:));
		end
	end
	% solve for the weights
	lambda = A \ Ftr;

	% what does the model predict at each of the training data
	Nin = size(Xin, 1);
	Fout = zeros(Nin,1);
	u = zeros(Ntr,K);
	for i=1:Nin
		for j=1:K
			u(i,j) = basisFunction(Xin(i,:) - C(j,:));
		end
		Fout(i) = u(i,:)*lambda;
	end

end

