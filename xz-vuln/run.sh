#! /bin/bash

set -eux -o pipefail

DATE=$(date +"%Y%m%d%H%M")
IMAGENAME=xz-vuln-be-careful-${DATE}
podman build -t ${IMAGENAME} .

set +e
echo "ld-linux-x86-64.so.2 version"
podman run -it ${IMAGENAME} /lib64/ld-linux-x86-64.so.2 --version

echo "Affected"
podman run -it ${IMAGENAME} time env -i LANG=C /usr/sbin/sshd -h
podman run -it ${IMAGENAME} env -i LANG=C /usr/sbin/sshd -h 2>&1 | tee /tmp/affected-${DATE}.log
ln -sf /tmp/affected-${DATE}.log /tmp/affected.log
echo "Not affected"
podman run -it ${IMAGENAME} time env -i LANG=C TERM=foo /usr/sbin/sshd -h 2>&1 | tee /tmp/not-affected-${DATE}.log
podman run -it ${IMAGENAME} env -i LANG=C TERM=foo /usr/sbin/sshd -h 2>&1 | tee /tmp/not-affected-${DATE}.log
ln -sf /tmp/not-affected-${DATE}.log /tmp/not-affected.log

podman run -it ${IMAGENAME} bash
