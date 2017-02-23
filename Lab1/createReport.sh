#!/bin/bash
# Runs MATLAB code to create images, and builds LaTeX report

matlab -nodesktop -nosplash -r "publish('labone.m', 'latex'); quit"

latexmk -pdf lab1.tex
