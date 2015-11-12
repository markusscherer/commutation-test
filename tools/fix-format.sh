#!/bin/sh

clang-format -i $1 2> /dev/null
astyle --options=./.astylerc --quiet $1
