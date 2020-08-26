# get all unit test sources
file(GLOB UNIT_TEST_SOURCES "${PROJECT_SOURCE_DIR}/testing/unit/*.cpp")

# create the tests
foreach(UNIT_TEST_CPP ${UNIT_TEST_SOURCES})
  get_filename_component(UNIT_TEST_NAME ${UNIT_TEST_CPP} NAME_WE)
  add_executable(${UNIT_TEST_NAME} ${UNIT_TEST_CPP})
  add_test(NAME ${UNIT_TEST_NAME} COMMAND ${UNIT_TEST_NAME})
endforeach()

# add properties to specific tests
set_tests_properties(failure PROPERTIES WILL_FAIL true)