CC=g++
CFLAGS=-g -std=c++11 -O3 -Wall -Wpedantic

all: bin/comtest bin/packer bin/random_functions

bin/comtest: src/comtest.cpp src/code_generators.hpp src/misc_tools.hpp src/constants.hpp src/primitive_solving_policy.hpp src/bitset_function.hpp
	$(CC) $(CFLAGS) src/comtest.cpp -o bin/comtest

bin/random_functions: src/random_functions.cpp src/bitset_function.hpp src/code_generators.hpp src/misc_tools.hpp src/constants.hpp
	$(CC) $(CFLAGS) src/random_functions.cpp -o bin/random_functions

bin/packer: src/packer.cpp src/bitset_function.hpp src/code_generators.hpp src/misc_tools.hpp src/constants.hpp
	$(CC) $(CFLAGS) src/packer.cpp -o bin/packer

cppcheck:
	rm -f .cppcheck-errors
	find src -name '*.[ch]pp' -execdir cppcheck --std=c++11 --enable=style,warning -q '{}' ';' 2> .cppcheck-errors
	[ ! -s .cppcheck-errors ]

check-format:
	rm -f .wrong-format
	find src -name '*.[ch]pp' -execdir ../tools/check-format.sh '{}' ';'
	[ ! -f .wrong-format ]

fix-format:
	find src -name '*.[ch]pp' -execdir ../tools/fix-format.sh '{}' ';'

generate-project-data:
	./tools/YCM-Generator/config_gen.py .

clean:
	rm -rf bin/*
