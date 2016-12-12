fileFormat = 'latex';

files = {'neuralNetwork.m', 'timeseriesPrediction.m', 'FinancialTimeSeries.m'};

for file = files
  publish(file{1}, fileFormat);
end
