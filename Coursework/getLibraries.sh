#!/bin/bash

# gets 5 years of daily S&P 500 data
wget -O SnP500.csv "http://chart.finance.yahoo.com/table.csv?s=^GSPC&a=11&b=11&c=2011&d=11&e=11&f=2016&g=d&ignore=.csv"

# gets MackeyGlass Code
mkdir ../libraries/
mkdir ../libraries/mackeyglass/
cd ../libraries/mackeyglass/ || exit

wget -O mackeyGlass.zip http://uk.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/24390/versions/1/download/zip
unzip mackeyGlass.zip

