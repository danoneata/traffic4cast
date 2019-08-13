#!/bin/bash

set -e
set -x

path=$1
name=$2

[[ -z "$path" ]] && { echo "No input path was given" ; exit 1; }
[[ -z "$name" ]] && { echo "No submission name was given" ; exit 1; }

for city in Berlin Istanbul Moscow; do
    dest=output/submissions/$city/${city}_test
    mkdir -p $dest
    cp -r $path/$city/* $dest
done

cd output/submissions
zip -r0 ${name}.zip .
rm -rf Berlin Istanbul Moscow
