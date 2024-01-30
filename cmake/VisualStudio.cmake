set(lab_benchmark_additional_files "" CACHE INTERNAL "")
set(lab_lib_additional_files "" CACHE INTERNAL "")
set(lab_test_additional_files "" CACHE INTERNAL "")

if(WIN32) # Benchmark
	list(APPEND lab_benchmark_additional_files "benchmark.h")
endif()
	
if(WIN32) # Lib
	list(APPEND lab_lib_additional_files "cuda/common.cuh")
	list(APPEND lab_lib_additional_files "cuda/encryption.cuh")
	list(APPEND lab_lib_additional_files "cuda/image.cuh")
	
	list(APPEND lab_lib_additional_files "encryption/Algorithm.h")
	list(APPEND lab_lib_additional_files "encryption/FES.h")
	list(APPEND lab_lib_additional_files "encryption/Key.h")
	
	list(APPEND lab_lib_additional_files "image/bitmap_image.h")
	list(APPEND lab_lib_additional_files "image/pixel.h")
	
	list(APPEND lab_lib_additional_files "io/image_parser.h")
	
	list(APPEND lab_lib_additional_files "util/Hash.h")
	
	list(APPEND lab_lib_additional_files "authors.h")
endif()

if(WIN32) # Tests
	list(APPEND lab_test_additional_files "test.h")	
	list(APPEND lab_test_additional_files "test_declarations.h")	
endif()
