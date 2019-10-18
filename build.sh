#!/bin/bash

# `./build`         - try to build the project
# `./build clear`   - clear CMake cache files
# `./build rebuild` - rebuild the whole project

function assert_errno() {
    if [ $1 -ne 0 ];
    then
        echo "ERROR: Some error occurs!"
        exit
    fi
}

function try_build() {
    echo "INFO: Trying to build project..."
    make

    if [ $? -ne 0 ];
    then
        echo "INFO: No make file exist, re-generate and try to build again..."
        cmake .
        assert_errno $?
        make
        assert_errno $?
    fi
    echo "INFO: built done!"
}

function clear_cache_at() {
    rm -rf $1"/"CMakeFiles/
    rm -rf $1"/"cmake_install.cmake
    rm -rf $1"/"Makefile
    rm -rf $1"/"CMakeCache.txt
}

function clear_cache() {
    echo "INFO: Cleaning cache files from $1/ ..."
    clear_cache_at $1
    for e in `ls $1`
    do
        full_name=$1"/"$e
        if [ -d $full_name ];
        then
            clear_cache $full_name
        fi
    done
    assert_errno $?
}

function try_rebuild() {
    echo "INFO: Tyring to rebuild the project..."
    clear_cache .
    assert_errno $?
    try_build
    assert_errno $?
}

if [ $# -ne 0 ];
then
    if [ $1 == "clear" ];
    then
        clear_cache .
    elif [ $1 == "rebuild" ];
    then
        try_rebuild
    else
        echo "INFO: Unrecognized parameter, ignore"
        try_build
    fi
else
    try_build
fi

echo "INFO: Everything is done!"
