#! /bin/bash
caffe train -solver solver.prototxt -gpu 1 2>&1 | tee log.txt
