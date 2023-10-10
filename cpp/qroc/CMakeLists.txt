set(CMAKE_CXX_FLAGS "-std=c++14 -O2")

set(master_equation_roc_SRCS
    master_equation_roc.cpp
)

#The reason for making "master_equation_roc" a SHARED library is that
#for the STATIC library, HIP cmake modules add the "-hc" compiler flag
#(heterogeneous compilation) to any executable that uses this library
#even if the executable in question does not have any heterogeneous
#code. The "-hc" flag causes issues when Eigen is included with enabled
#vectorization. Eigen's use of intrinsics causes HCC to fail with something like
# always_inline function '_mm_loadu_pd' requires target feature 'sse2', but would be inlined into function 'ploadu' that is compiled without support for 'sse2'
#Apparently, it is expected behaviour
#https://github.com/RadeonOpenCompute/hcc/issues/976
#On the other hand "-hc" flag is not added 
#for SHARED libary. This is why we use SHARED.

add_library(master_equation_roc SHARED ${master_equation_roc_SRCS})
target_link_libraries(master_equation_roc PRIVATE roc::rocsparse hip::host)
