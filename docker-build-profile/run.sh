#! /bin/bash
set -eux -o pipefail

strace \
    --trace=execve,execveat,exit,exit_group \
    --follow-forks \
    --string-limit=1000 \
    --absolute-timestamps=format:unix,precision:us \
    --output=straceprof.log \
    podman build . --no-cache
straceprof \
    --log=straceprof.log \
    --output=straceprof.png
