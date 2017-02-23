#!/bin/bash
# Runs MATLAB code to create images, and builds LaTeX report

matlab -nodesktop -nosplash -r "publish('labthree.m', 'latex'); quit"

latexmk -pdf lab3.tex
