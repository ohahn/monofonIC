cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    genericio
    GIT_REPOSITORY https://git.cels.anl.gov/hacc/genericio.git
    GIT_TAG master
    GIT_SHALLOW YES
    GIT_PROGRESS TRUE
    USES_TERMINAL_DOWNLOAD TRUE   # <---- this is needed only for Ninja
)

FetchContent_GetProperties(genericio)
if(NOT genericio_POPULATED)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Populate(genericio)
    add_subdirectory(${genericio_SOURCE_DIR} ${genericio_BINARY_DIR})
endif()
