#! /bin/bash

set -eux -o pipefail

podman build -t xz-vuln-be-careful .
podman run -it xz-vuln-be-careful:latest bash
