cmake_minimum_required(VERSION 3.16)
project(opencl_apu_vision_acceleration LANGUAGES CXX)

# indichiamo al compilatore di usare C++17 e rendiamolo un requisito obbligatorio 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# linking alle librerie esterne
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL ${OpenCV_LIBS})
target_link_libraries(benchmark PRIVATE OpenCL::OpenCL ${OpenCV_LIBS})

# copia dell'intera cartella "kernels" dalla cartella corrente alla cartella di compilazione
# in questo modo è possibile compilare i file sorgenti OpenCL a runtime 
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kernels/)
