include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(vdb_engine_tests
    test_main.cpp
    test_distance.cpp
    test_vector_store.cpp
    test_search.cpp
    test_ivf_index.cpp
)
target_link_libraries(vdb_engine_tests PRIVATE vdb_engine doctest::doctest)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(vdb_engine_tests)
