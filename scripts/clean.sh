#!/bin/bash

echo Removing all .c, .so and .html files...

find cherab -type f -name '*.c' -exec rm -rf {} +
find cherab -type f -name '*.so' -exec rm -rf {} +
find cherab -type f -name '*.html' -exec rm -rf {} +
rm build -rf
