add_executable(simple_clustering.out simple_clustering.cpp)
target_link_libraries(simple_clustering.out PRIVATE superkmeans)

add_executable(hierarchical_clustering.out hierarchical_clustering.cpp)
target_link_libraries(hierarchical_clustering.out PRIVATE superkmeans)

# Custom target to build all examples with 'make examples'
add_custom_target(examples
    DEPENDS
        simple_clustering.out
        hierarchical_clustering.out
)
