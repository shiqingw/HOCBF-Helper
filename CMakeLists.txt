cmake_minimum_required(VERSION 3.18..3.20)

project(HOCBFHelperPy VERSION 0.0.1)

# set(CMAKE_BUILD_TYPE Debug)  # Enable debug symbols
# set(CMAKE_EXE_LINKER_FLAGS "-fsanitize=address")

find_package(OpenBLAS REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(xsimd REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(scs REQUIRED)
find_package(scalingFunctionsHelper REQUIRED)

option(BUILD_PYTHON_MODULE "Build the python model" OFF)

if(BUILD_PYTHON_MODULE)
    pybind11_add_module(
        HOCBFHelperPy
        src/pybind.cpp
        src/utils.cpp
        src/threadPool.cpp
        src/problem3d.cpp
        src/problem3dCollection.cpp
        src/problem3dCollectionMovingObstacle.cpp
        src/ellipsoidAndLogSumExp3dPrb.cpp
        src/ellipsoidAndHyperplane3dPrb.cpp
        src/ellipsoidAndEllipsoid3dPrb.cpp
        src/problem2d.cpp
        src/problem2dCollection.cpp
        src/ellipsoidAndLogSumExp2dPrb.cpp
        src/ellipsoidAndHyperplane2dPrb.cpp
    )

    if(EXISTS /usr/local/include)
        # Add /usr/local/include to the include directories for your target
        target_include_directories(HOCBFHelperPy PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}  # Include this if find_package sets this variable
            /usr/local/include
        )
    else()
        target_include_directories(HOCBFHelperPy PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}  # Include this if find_package sets this variable
        )
    endif()

    # If OpenBLAS provides a target, use it directly. If not, specify the library path as you did.
    # set(SCS_DIR_LIBRARIES "/usr/local/lib/")
    target_link_libraries(HOCBFHelperPy PUBLIC
        pybind11::module
        xtensor
        xtensor::optimize
        # xsimd # Do not use this, code gets slower
        # xtensor::use_xsimd # Do not use this, code gets slower
        xtensor-python
        Python::NumPy
        ${OpenBLAS_LIBRARIES}  # Use the variable set by find_package if available
        scs::scsdir
        scalingFunctionsHelper::scalingFunctionsHelper
        pthread
        m
        dl
    )

    target_compile_definitions(HOCBFHelperPy PRIVATE VERSION_INFO=0.0.1)
    set_property(TARGET HOCBFHelperPy PROPERTY CXX_STANDARD 14)

else()

    add_executable(main 
        src/main.cpp
        src/threadPool.cpp
        src/utils.cpp
        src/problem3d.cpp
        src/problem3dCollection.cpp
        src/problem3dCollectionMovingObstacle.cpp
        src/ellipsoidAndLogSumExp3dPrb.cpp
        src/ellipsoidAndHyperplane3dPrb.cpp
        src/ellipsoidAndEllipsoid3dPrb.cpp
        src/problem2d.cpp
        src/problem2dCollection.cpp
        src/ellipsoidAndLogSumExp2dPrb.cpp
        src/ellipsoidAndHyperplane2dPrb.cpp
    )

    if(EXISTS /usr/local/include)
        # Add /usr/local/include to the include directories for your target
        target_include_directories(main PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}  # Include this if find_package sets this variable
            /usr/local/include
        )
    else()
        target_include_directories(main PUBLIC
            ${OpenBLAS_INCLUDE_DIRS}  # Include this if find_package sets this variable
        )
    endif()

    # If OpenBLAS provides a target, use it directly. If not, specify the library path as you did.
    target_link_libraries(main PUBLIC
        pybind11::module
        xtensor
        xtensor::optimize
        # xsimd # Do not use this, code gets slower
        # xtensor::use_xsimd # Do not use this, code gets slower
        xtensor-python
        Python::NumPy
        ${OpenBLAS_LIBRARIES}  # Use the variable set by find_package if available
        scs::scsdir
        scalingFunctionsHelper::scalingFunctionsHelper
        pthread
        m
        dl
    )

    target_compile_definitions(main PRIVATE VERSION_INFO=0.0.1)
    set_property(TARGET main PROPERTY CXX_STANDARD 14)

endif()
