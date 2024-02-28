#!/bin/bash

while IFS= read -r line; do
    cd $2
    wget $line
done < "$1"

## sh download.sh ./qwen/test.txt ./qwen/
## sh download.sh ./qwen/files.txt ./qwen/