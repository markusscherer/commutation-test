#!/bin/sh

conf_dir=$(dirname $1)
name=comtest_$(basename $conf_dir)

outdir=bin/

if [[ -z $CC ]]; then
  CC=g++
fi

options='-std=c++11 -Wall -Wpedantic -msse4.1 -O3'

$CC -I src/ -I $conf_dir $2 src/comtest.cpp $options -o $outdir$name
