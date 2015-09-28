#!/bin/bash
grep '^v' | # get correct line
tr " " "\n" | grep -v "^0$" | egrep '^[[:digit:]]' | # get numbers
xargs -I _ expr _ - 1 | xargs -I _ echo -e + \\n 2 ^ _ | # prepare calculation 
sed '1d' | # drop first plus
tr "\n" " " | sed '$a\' | #remove and add newlines where needed
bc

# quite hacky ;-(
