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

include($$(MY_LIB_PATH)/QtOpenCL/QtOpenCL_libs.pri)

HEADERS +=
SOURCES += main.cpp

RESOURCES += resources.qrc
