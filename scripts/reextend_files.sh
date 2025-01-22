#!/usr/bin/env bash

find "$1" -type f \( -name "*.rule" -o -name "*.rule.json" \) -exec sh -c '
    for file; do
        if [[ "$file" == *.rule.json ]]; then
            mv -- "$file" "${file%.rule.json}.pl.json"
        elif [[ "$file" == *.rule ]]; then
            mv -- "$file" "${file%.rule}.pl"
        fi
    done
' _ {} +