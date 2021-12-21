cmake_minimum_required(VERSION 3.16)
project(SymbolStrings)


set(
	SOURCES
	src/raw.cpp
	src/labels.cpp
	src/entity-linking.cpp
	src/fix-details.cpp
	src/entity-masking.cpp
	src/suffix-tree/unicode-string.cpp
	src/suffix-tree/explicit-state.cpp
	src/suffix-tree/suffix-tree.cpp
	src/suffix-tree/longest-common-substring.cpp
	src/main.cpp
)


add_executable(symbol-strings ${SOURCES})
target_include_directories(symbol-strings PRIVATE inc)


add_subdirectory(utf-cpp)
target_link_libraries(symbol-strings PRIVATE utf8cpp)

find_package(unofficial-curlpp CONFIG REQUIRED)
target_link_libraries(symbol-strings PRIVATE unofficial::curlpp::curlpp)


find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(symbol-strings PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)


target_compile_options(symbol-strings PUBLIC -std=c++17 -Wall -pedantic -g)

