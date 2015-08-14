#!/bin/sh

orig=$(md5sum $1 | cut -f 1 -d ' ')
tmpfile=$(mktemp)

clang-format $1 2> /dev/null | astyle --options=../.astylerc > $tmpfile
#clang-format $1 > $tmpfile

formatted=$(md5sum $tmpfile | cut -f 1 -d ' ')

if [ $orig = $formatted ]
then
  rm -rf $tmpfile
else
  git diff $1 $tmpfile
  rm -rf $tmpfile
  echo $1 >> ../.wrong-format
  exit 1
fi
