TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../src/main.cpp \
    ../src/tools.cpp \
    ../src/ukf.cpp

HEADERS += \
    ../src/ground_truth_package.h \
    ../src/measurement_package.h \
    ../src/tools.h \
    ../src/ukf.h
