function [ trainData, testData ] = convertTimeSeriesToMLInput( data, time, p, train_range, test_range )
%convertTimeSeriesToMLInput Converts a time series to input/output data
%   Converts a time series to training input/output data and testing
%   input/ouptut data. 
%   For example, converts [1 2 3 4 5 6 7 8 9 10] to (if p = 5):
%    input:        output:
%   [ 1 2 3 4 5;   [ 6;
%     2 3 4 5 6;     7;
%     3 4 5 6 7;     8;
%     4 5 6 7 8;     9;
%     5 6 7 8 9 ]    10 ]

% tr_tst(1) contains training data
% tr_tst(2) contains testing data
tr_tst(1).range = train_range;
tr_tst(2).range = test_range;

% for training or testing
for tr_or_tst = [1 2]
	
	% data_size is the size of the data
	% used to cut down the size of the array at the end
	data_size = 0;
	
	max_length = length(tr_tst(tr_or_tst).range);
	tr_tst(tr_or_tst).data.in = zeros(max_length, p);
	tr_tst(tr_or_tst).data.out = zeros(max_length, 1);
	tr_tst(tr_or_tst).data.time = zeros(max_length, 1);
	
	for i = tr_tst(tr_or_tst).range
		if(i <= p)
			% can't do anything since too close to beginning of the array
		else
			data_size = data_size + 1;
			% input p data values
			tr_tst(tr_or_tst).data.in(data_size,:) = data( (i - p): (i - 1));
			% output data value
			tr_tst(tr_or_tst).data.out(data_size) = data(i);
			% time of the output
			tr_tst(tr_or_tst).data.time(data_size) = time(i);
		end
	end
	
	% cuts the unused rows of the matrices out
	tr_tst(tr_or_tst).data.in   = tr_tst(tr_or_tst).data.in(1:data_size,:);
	tr_tst(tr_or_tst).data.out  = tr_tst(tr_or_tst).data.out(1:data_size);
	tr_tst(tr_or_tst).data.time = tr_tst(tr_or_tst).data.time(1:data_size);
end

trainData = tr_tst(1).data;
testData  = tr_tst(2).data;

end