#!/bin/bash
# The following script gets the necessary libraries, runs all the matlab code to
# create the images for the LaTeX code, then compiles the LaTeX to create the
# report

# gets and installs the libraries
./getLibraries.sh

# runs all the code to create images for LaTeX
matlab -nodesktop -nosplash < createImages.m

# compiles the TeX file and makes a .pdf
# use latexmk so that bibtex and hyperlinks compile properly
latexmk -pdf Coursework.tex
