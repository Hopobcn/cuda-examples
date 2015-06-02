
# Download and install GoogleTest
ExternalProject_Add(
    gtest
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    #--Download step-------------
    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
    URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
    URL_HASH SHA1=f85f6d2481e2c6c4a18539e391aa4ea8ab0394af
    #--Update/Patch step----------
    #--Configure step-------------
    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
               -Dgtest_force_shared_crt=ON
    #--Build step-----------------
    #--Install step--------------- (DISABLED)
    INSTALL_COMMAND ""
    #--Test step------------------
    #--Output logging-------------
    #--Custom targets-------------
)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set gtest properties
ExternalProject_Get_Property(gtest SOURCE_DIR BINARY_DIR)
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION"                 "${BINARY_DIR}/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
    #"INTERFACE_INCLUDE_DIRECTORIES"     "${SOURCE_DIR}/include"
)
# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${SOURCE_DIR}/include")

set(GTEST_INCLUDE_DIRS ${SOURCE_DIR}/include PARENT_SCOPE)
