#-------------------------------------------------
#
# Project created by QtCreator 2014-11-14T16:45:12
#
#-------------------------------------------------

QT += core
QT -= gui

TARGET = OpenCL_local_memory
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -g

include($$(MY_LIB_PATH)/QtOpenCL/QtOpenCL_libs.pri)

HEADERS += \
    input.h
SOURCES += main.cpp \
    input.cpp

RESOURCES += resources.qrc
