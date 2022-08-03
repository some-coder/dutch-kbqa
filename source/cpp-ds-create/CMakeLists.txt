cmake_minimum_required(VERSION 3.16)

set(CONCAT DESCRIPTION_VAR "Symbols for creating a Dutch derivation "
                           "of the LC-QuAD 2.0 dataset.")
project(DutchKBQACPPDSCreate
        VERSION "0.1.0"
        DESCRIPTION "${DESCRIPTION_VAR}"
        HOMEPAGE_URL "https://github.com/some-coder/dutch-kbqa"
        LANGUAGES "CXX")

file(GLOB_RECURSE SOURCES_VAR "source/*.cpp")
message("Obtained the following source files: \"${SOURCES_VAR}\".")

add_executable(main "${SOURCES_VAR}")
target_include_directories(main PRIVATE "include/")

find_package(utf8cpp CONFIG REQUIRED)
target_link_libraries(main PRIVATE utf8cpp)
find_package(unofficial-curlpp CONFIG REQUIRED)
target_link_libraries(main PRIVATE unofficial::curlpp::curlpp)
find_package(jsoncpp CONFIG REQUIRED)
target_link_libraries(main PRIVATE jsoncpp_object
                                   jsoncpp_static
                                   JsonCpp::JsonCpp)
find_package(Boost REQUIRED COMPONENTS program_options)
target_link_libraries(main PRIVATE Boost::boost Boost::program_options)

if (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR
    ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android") OR
		("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD"))
	# Assume the GNU Compiler Collection (GCC).
	target_compile_options(main PUBLIC "-std=c++17"
	                                   "-Wall"
	                                   "-pedantic"
	                                   "-g")
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
	# Assume Clang.
	target_compile_options(main PUBLIC "-std=c++17"
	                                   "-Wall"
	                                   "-pedantic"
	                                   "-g")
elseif(("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows") OR
       ("${CMAKE_SYSTEM_NAME}" STREQUAL "MSYS"))
	# Assume the Microsoft Visual Studio (MSVC) compiler.
	target_compile_options(main PUBLIC "/std:c++17"
	                                   "/Wall"
	                                   "/Zi")
else()
	# Panic on other platforms.
	message(FATAL_ERROR "Unsupported platform \"${CMAKE_SYSTEM_NAME}\".")
endif()

