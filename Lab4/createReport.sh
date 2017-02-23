#!/bin/bash
# Runs MATLAB code to create images, and builds LaTeX report

matlab -nodesktop -nosplash -r "publish('lab4.m', 'latex'); quit"

latexmk -pdf lab4.tex
