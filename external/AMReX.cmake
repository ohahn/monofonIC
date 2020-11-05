cmake_minimum_required(VERSION 3.11)
include(FetchContent)
FetchContent_Declare(
    amrex
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
    GIT_TAG development
    GIT_SHALLOW YES
    GIT_PROGRESS TRUE
    USES_TERMINAL_DOWNLOAD TRUE   # <---- this is needed only for Ninja
)

FetchContent_GetProperties(amrex)
if(NOT amrex)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Populate(amrex)
    add_subdirectory(${amrex_SOURCE_DIR} ${amrex_BINARY_DIR})
endif()