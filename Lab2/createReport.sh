#!/bin/bash
# Runs MATLAB code to create images, and builds LaTeX report

matlab -nodesktop -nosplash -r "publish('labtwo.m', 'latex'); quit"

latexmk -pdf lab2.tex
