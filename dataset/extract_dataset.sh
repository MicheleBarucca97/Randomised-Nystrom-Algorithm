#!/usr/bin/env bash
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2
bzip2 -d mnist.bz2
bzip2 -d mnist.t.bz2
head -n 8192 mnist > mnist_not_scaled
