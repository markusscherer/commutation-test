# commutation-test

This is the code I wrote for my diploma thesis "Parallelizing the Commutation
Property for Functions over Small Domains". 

The code is quite *academic* and by this I mean has rough edges.

If you want to use this program for your research but have problems: shoot me a
message!

```
     Here be dragons!                           __----~~~~~~~~~~~------___
                                      .  .   ~~//====......          __--~ ~~
                      -.            \_|//     |||\\  ~~~~~~::::... /~
                   ___-==_       _-~o~  \/    |||  \\            _/~~-
           __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
       _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
     .~       .~       |   \\ -_    /  /-   /   ||      \   /
    /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
    |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
             '         ~-|      /|    |-~\~~       __--~~
                         |-~~-_/ |    |   ~\_   _-~            /\
                              /  \     \__   \/~                \__
                          _--~ _/ | .-~~____--~-/                  ~~==.
                         ((->/~   '.|||' -_|    ~~-/ ,              . _||
                                    -_     ~\      ~~---l__i__i__i--~~_/
                                    _-~-__   ~)  \--______________--~~
                                  //.-~~~-~_--~- |-------~~~~~~~~
                                         //.-~~~--\
```

## Structure of the Code

There are different programs which can be compiled to an executable:

| Filename                          | comment               | domain size | arities | 
| --------------------------------- | --------------------- | ----------- | --------|
|`src/comtest.cpp`                  | default program       |           4 | 1,2,3,4 |
|`src/comtest.cu`                   | gpu program           |           4 |       4 |
|`src/comtest3.cpp`                 | special encoding (\*) |           3 |   1,2,3 |
|`src/primitive_comtest3.cpp`       | non-configurable      |           3 | 1,2,3,4 |
|`src/primitive_comtest.cpp`        | non-configurable      |           2 | 1,2,3,4 |

(\*) This executable take functions over a three-element domains encoded as
functions over


The programs not marked as "non-configurable" include a header
`comtest.config.hpp` which does not lie in the standard code directory `src`.
We therefore have to specify an include path where a file called
`comtest.config.hpp` lies.
This file has to make two classes visible:

```c++
template <uint64_t D, uint64_t A1, uint64_t A2> struct solver {
    static inline bool commutes(const array_function<D, A1, ElementType>& f1,
                                const array_function<D, A2, ElementType>& f2);
};
```

and

```c++
struct tester {
    typedef some_type matches_type;

    // configure tester
    static void init(const std::map<std::string, std::string>&); 

    template <uint64_t D, uint64_t A1, uint64_t A2>
    static inline matches_type
    commutation_test(const std::vector<array_function<D, A1, uint8_t>>& vec1,
                     const std::vector<array_function<D, A2, uint8_t>>& vec2);
};
```

The `solver` checks one particular pair of functions for commutation, the
`tester` checks vectors of functions for commutation (and will usually call
the `solver`). `tester::commutation_test` returns an instance of `matches_type`
which has be defined in `tester`. Right now only one `matches_type` based on
`std::map` is supported.

Some programs use a function type different from `array_function`.

Examples for configurations are in the `src/configs/` directory.

You can compile binaries as in the following examples:

```bash
# instead of -msse4.1 you can use -mavx2 if it is available on your target
# architecture (this is needed to enable the SIMD intrinsics)
g++ -std=c++11 -I src/ -I src/configs/selective_selective src/comtest.cpp -Wall -Wpedantic -msse4.1 -O3 

# some policies require special libraries (here pthreads)
clang++ -std=c++11 -lpthread -I src/ -I src/configs/multi_threaded/ src/comtest.cpp -Wall -Wpedantic -msse4.1 -O3

# some policies require special compiler flags
g++ -std=c++11 -fopenmp -I src/ -I src/configs/openmp_critical src/comtest.cpp -Wall -Wpedantic -msse4.1 -O3

# comtest.cu even requires a special compiler
/opt/cuda/bin/nvcc -arch=compute_35 -code=sm_35 -lcuda -O3 -std=c++11 -I src/ -I src/configs/cuda_config src/comtest.cu
```

All binaries take the input as binary files of densely packed functions,
specified via the command line switches `-1`, `-2-`, `-3`, `-4` (respectively
`--functions-1` and so on). For each arity only one file may be specified. All
supplied functions get checked with each other. The format of the input  is
discussed further below.

Each binary may output debugging information to `stderr`.

The output of each binary (to `stdout`) looks as follows:

```
<number of following lines>
<function id>/<arity>: <id of first commuting function>/<arity> <id of second commuting function>/<arity>  (and so on)
<function id>/<arity>: <id of first commuting function>/<arity> <id of second commuting function>/<arity>  (and so on)
```

The function id has nothing to do with the function itself but is the
position of the function in the supplied input file.

## Input Format
The commutation checkers take binary files as inputs were every line in the
result row of the function takes 1 bit (domain size 2) or 2 bits (domain sizes 3
and 4). The least significant bits encode functions with lower arguments, the
most significant with high arguments. The space each function takes is rounded
up to the next multiple of eight bits.

The following function corresponds to the hex-value `e0f9845e` (given by `xxd -c4 -e`)

```
0 0 2
0 1 3
0 2 1
0 3 1
1 0 0
1 1 1
1 2 0
1 3 2
2 0 1
2 1 2
2 2 3
2 3 3
3 0 0
3 1 0
3 2 2
3 3 3
```

To make things easier, a binary `bin/packer` is provided in this repository.
Use `make bin/packer` to build.

`packer` supports two input formats. The one above, where every line in
explicitly given (order does not matter) and a more compact one (enabled with
the `-c` command line switch).

All text input files have to have the following preamble:

```
count <number of functions>
domain_size 4
arity 4
```

Giving the wrong number of functions will cause subtle and not so subtle bugs.

When using the first format all values are initialized with 3 (respectively 1
for domain size 2) and function definitions can be finished with `end`. This is
useful when encoding functions over domain size 3 as functions over domain size
4 (for an example see `tests/data/e4_all_functions.3.2`). If no `end` statement
is encountered, the next functions is assumed to start after `domain_size^arity`
lines have been read.

In the compact format the function from above would look like this:

```
2311010212330023
```

In this format one function is given per line.

For more examples look into the data in `tests/data`.

To find out how to use `packer` exactly, execute `packer -h`.
