TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    ../src/main.cpp \
    ../src/PID.cpp

HEADERS += \
    ../src/json.hpp \
    ../src/PID.h

INCLUDEPATH += \
    /usr/local/include \
    /usr/local/Cellar/openssl/1.0.2k/include
