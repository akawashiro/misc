#! /bin/bash
set -eux -o pipefail

script_dir=$(dirname "$(readlink -f "$0")")
qemu_ubuntu_dir=/tmp/qemu-ubuntu

if ! command -v cloud-localds >/dev/null 2>&1; then
    sudo apt-get install --yes cloud-image-utils --fix-missing
fi
if ! command -v qemu-system-x86_64 >/dev/null 2>&1; then
    sudo apt-get install --yes qemu-system-x86 --fix-missing
fi
if ! command -v qemu-img >/dev/null 2>&1; then
    sudo apt-get install --yes qemu-utils --fix-missing
fi

mkdir -p "${qemu_ubuntu_dir}"
cd "${qemu_ubuntu_dir}"

if [ ! -f noble-server-cloudimg-amd64.img ]; then
    curl -fLO https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img
fi

rm -rf vm.qcow2
if [ ! -f vm.qcow2 ]; then
    qemu-img create -f qcow2 -F qcow2 -b noble-server-cloudimg-amd64.img vm.qcow2
fi

rm -rf seed.img
cloud-localds seed.img "${script_dir}/user-data" "${script_dir}/meta-data"

qemu-system-x86_64 \
  -machine type=q35 \
  -cpu qemu64 \
  -m 2G \
  -smp 2 \
  -drive file=vm.qcow2,format=qcow2,if=virtio \
  -drive file=seed.img,format=raw,if=virtio \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -nographic \
  -serial mon:stdio \
  -accel tcg
