
# Download and install GoogleMock
ExternalProject_Add(
    gmock
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    #--Download step-------------
    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
    URL https://googlemock.googlecode.com/files/gmock-1.7.0.zip

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

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gmock)

# Set gmock properties
ExternalProject_Get_Property(gmock SOURCE_DIR BINARY_DIR)
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION"                 "${BINARY_DIR}/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
#    "INTERFACE_INCLUDE_DIRECTORIES"    "${SOURCE_DIR}/include"
)
# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${SOURCE_DIR}/include")

set(GMOCK_INCLUDE_DIRS ${SOURCE_DIR}/include PARENT_SCOPE)