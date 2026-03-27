cmake_minimum_required(VERSION 3.26)
project(superkmeans_fetchcontent_test)

include(FetchContent)

FetchContent_Declare(superkmeans
    SOURCE_DIR ${SUPERKMEANS_SOURCE_DIR}
)
FetchContent_MakeAvailable(superkmeans)

add_executable(fetchcontent_test main.cpp)
target_link_libraries(fetchcontent_test PRIVATE superkmeans)
