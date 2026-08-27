cmake_minimum_required(VERSION 3.16)
project(opencl_apu_vision_acceleration LANGUAGES CXX)

# indichiamo al compilatore di usare C++17 e rendiamolo un requisito obbligatorio 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release come build type di default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

# troviamo le dipendenze esterne 
find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

# raggruppa i file sorgente comuni ai due eseguibili in una variabile SOURCES 
set(SOURCES src/OpenCLManager.cpp src/AlgoritmiCPU.cpp)

# creazione del target eseguibile con menu interattivo
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCES})

# creazione del target eseguibile di benchmark
add_executable(benchmark src/benchmark.cpp ${SOURCES})

# indicazioni per trovare i file header .hpp
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang") # se compilatore è gcc o clang applica ottimizzazioni 
    
    set(FLAGS
        -O3                 # massima ottimizzazione del compilatore
        -march=native       # abilita le istruzioni SIMD specifiche della APU
        -ffast-math         # ottimizza i calcoli float 
        -flto               # link time optimization, consente l'ottimizzazione dell'intero programma durante la fase di linking 
    )

    # l'uso della generator expression $<condizione:valore>, se condizione vera viene restituito il valore
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:${FLAGS}>) 
    target_compile_options(benchmark PRIVATE $<$<CONFIG:Release>:${FLAGS}>)

    target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-flto>)
    target_link_options(benchmark PRIVATE $<$<CONFIG:Release>:-flto>)
endif()

# linking alle librerie esterne
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL ${OpenCV_LIBS})
target_link_libraries(benchmark PRIVATE OpenCL::OpenCL ${OpenCV_LIBS})

# copia dell'intera cartella "kernels" dalla cartella corrente alla cartella di compilazione
# in questo modo è possibile compilare i file sorgenti OpenCL a runtime 
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kernels/)
