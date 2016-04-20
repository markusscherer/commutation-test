#!/bin/sh

conf_dir=$(dirname $1)

outdir=bin/

if [[ -z $CC ]]; then
  CC=g++
fi
name=$CC""_comtest_$(basename $conf_dir)


options='-std=c++11 -Wall -Wpedantic -msse4.1 -O3'

$CC -I src/ -I $conf_dir $2 src/comtest.cpp $options -o $outdir$name
