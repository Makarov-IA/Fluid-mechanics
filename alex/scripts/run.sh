#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ALEX_ROOT="$SCRIPT_DIR/.."

cd "$ALEX_ROOT" || exit 1

START_TIME=$(date +%s.%N)

make run

END_TIME=$(date +%s.%N)
DURATION=$(awk "BEGIN {print $END_TIME - $START_TIME}")
printf "\nОбщее время: %.3f с\n" "$DURATION"
