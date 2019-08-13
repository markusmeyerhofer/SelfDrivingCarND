TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../src/FusionEKF.cpp \
    ../src/kalman_filter.cpp \
    ../src/main.cpp \
    ../src/tools.cpp

HEADERS += \
    ../src/FusionEKF.h \
    ../src/ground_truth_package.h \
    ../src/kalman_filter.h \
    ../src/measurement_package.h \
    ../src/tools.h
