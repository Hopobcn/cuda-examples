
FILE(GLOB sourcefiles "../src/*.cpp" "../src/*.cu")
include_directories("../include" SYSTEM ${INCLUDE_DIRS})

cuda_add_library(cuda_examples_lib STATIC ${sourcefiles})