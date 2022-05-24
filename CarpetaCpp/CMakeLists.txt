cmake_minimum_required(VERSION 2.8)

project(Parcial3HPC)
set(project_sources
    main.cpp
    RegresionLineal/regresionlineal.cpp
    Extraer/extraer.cpp)

set(project_headers
    RegresionLinear/regresionlineal.h
    Extraer/extraer.h)

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})
