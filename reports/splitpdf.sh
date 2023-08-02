#!/usr/bin/env sh

for filename in CompareEnzUse_*.pdf; do
    echo "$filename"
    pdfseparate "$filename" "$(basename "$filename" .pdf)_%d.pdf"
done
