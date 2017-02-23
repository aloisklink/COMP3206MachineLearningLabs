#!/bin/bash
# Runs MATLAB code to create images, and builds LaTeX report

matlab -nodesktop -nosplash -r "publish('lab5.m', 'latex'); quit"

latexmk -pdf lab5.tex
