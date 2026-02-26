#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TASK2_ROOT="$SCRIPT_DIR/.."

cd "$TASK2_ROOT" || exit 1

START_TIME=$(date +%s.%N)

make all || exit 1
./task2 "$@"
STATUS=$?

END_TIME=$(date +%s.%N)
DURATION=$(awk "BEGIN {print $END_TIME - $START_TIME}")
printf "\nTotal time: %.3f s\n" "$DURATION"

exit $STATUS
