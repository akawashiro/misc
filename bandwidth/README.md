# Bandwidth benchmarks

## Machine information

```
$ sudo lshw -class memory
  *-firmware
       description: BIOS
       vendor: American Megatrends International, LLC.
       physical id: 0
       version: P2.10
       date: 08/04/2021
       size: 64KiB
       capacity: 16MiB
       capabilities: pci upgrade shadowing cdboot bootselect socketedrom edd int13floppynec int13floppytoshiba int13floppy360 int13floppy1200 int13floppy720 int13floppy2880 int5printscreen int9keyboard int14serial int17printer int10video acpi usb biosbootspecification uefi
  *-memory
       description: System Memory
       physical id: 10
       slot: System board or motherboard
       size: 64GiB
     *-bank:0
          description: [empty]
          product: Unknown
          vendor: Unknown
          physical id: 0
          serial: Unknown
          slot: DIMM 0
     *-bank:1
          description: DIMM DDR4 Synchronous Unbuffered (Unregistered) 3200 MHz (0.3 ns)
          product: CT32G4DFD832A.C16FE
          vendor: Unknown
          physical id: 1
          serial: E60BBC35
          slot: DIMM 1
          size: 32GiB
          width: 64 bits
          clock: 3200MHz (0.3ns)
     *-bank:2
          description: [empty]
          product: Unknown
          vendor: Unknown
          physical id: 2
          serial: Unknown
          slot: DIMM 0
     *-bank:3
          description: DIMM DDR4 Synchronous Unbuffered (Unregistered) 3200 MHz (0.3 ns)
          product: CT32G4DFD832A.C16FE
          vendor: Unknown
          physical id: 3
          serial: E60BC28D
          slot: DIMM 1
          size: 32GiB
          width: 64 bits
          clock: 3200MHz (0.3ns)
  *-cache:0
       description: L1 cache
       physical id: 13
       slot: L1 - Cache
       size: 1MiB
       capacity: 1MiB
       clock: 1GHz (1.0ns)
       capabilities: pipeline-burst internal write-back unified
       configuration: level=1
  *-cache:1
       description: L2 cache
       physical id: 14
       slot: L2 - Cache
       size: 8MiB
       capacity: 8MiB
       clock: 1GHz (1.0ns)
       capabilities: pipeline-burst internal write-back unified
       configuration: level=2
  *-cache:2
       description: L3 cache
       physical id: 15
       slot: L3 - Cache
       size: 64MiB
       capacity: 64MiB
       clock: 1GHz (1.0ns)
       capabilities: pipeline-burst internal write-back unified
       configuration: level=3
```

## Benchmark results using existing tools

### memcpy

```
$ perf bench mem memcpy --size 1GB
# Running 'mem/memcpy' benchmark:
# function 'default' (Default memcpy() provided by glibc)
# Copying 1GB bytes ...

      17.304930 GB/sec
# function 'x86-64-unrolled' (unrolled memcpy() in arch/x86/lib/memcpy_64.S)
# Copying 1GB bytes ...

       9.950447 GB/sec
# function 'x86-64-movsq' (movsq-based memcpy() in arch/x86/lib/memcpy_64.S)
# Copying 1GB bytes ...

      19.133263 GB/sec
```

### TCP

#### localhost

```
$ iperf3 -c localhost --format g -bytes 1g
Connecting to host localhost, port 5201
[  5] local 127.0.0.1 port 59876 connected to 127.0.0.1 port 5201
[ ID] Interval           Transfer     Bitrate         Retr  Cwnd
[  5]   0.00-1.00   sec  4.83 GBytes  4.82 GBytes/sec    0   6.56 MBytes       
[  5]   1.00-2.00   sec  3.09 GBytes  3.10 GBytes/sec    0   6.56 MBytes       
[  5]   2.00-3.00   sec  1.84 GBytes  1.84 GBytes/sec    0   6.56 MBytes       
[  5]   3.00-4.00   sec  1.90 GBytes  1.90 GBytes/sec    0   6.56 MBytes       
[  5]   4.00-5.00   sec  4.30 GBytes  4.30 GBytes/sec    0   6.56 MBytes       
[  5]   5.00-6.00   sec  5.37 GBytes  5.37 GBytes/sec    0   6.56 MBytes       
[  5]   6.00-7.00   sec  5.00 GBytes  5.00 GBytes/sec    0   6.56 MBytes       
[  5]   7.00-8.00   sec  4.23 GBytes  4.23 GBytes/sec    0   6.56 MBytes       
[  5]   8.00-9.00   sec  4.17 GBytes  4.17 GBytes/sec    0   6.56 MBytes       
[  5]   9.00-10.00  sec  5.35 GBytes  5.35 GBytes/sec    0   6.56 MBytes       
- - - - - - - - - - - - - - - - - - - - - - - - -
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-10.00  sec  51.8 GBytes  5.18 GBytes/sec    0             sender
[  5]   0.00-10.00  sec  51.8 GBytes  5.18 GBytes/sec                  receiver
```

#### LAN network (masumi <-> WSL)

```
$ iperf3 -c 192.168.11.70 --format G -bytes 1G
Connecting to host 192.168.11.70, port 5201
[  5] local 172.20.125.7 port 60718 connected to 192.168.11.70 port 5201
[ ID] Interval           Transfer     Bitrate         Retr  Cwnd
[  5]   0.00-1.00   sec   114 MBytes  0.11 GBytes/sec    0   3.95 MBytes
[  5]   1.00-2.00   sec   112 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   2.00-3.00   sec   110 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   3.00-4.00   sec   111 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   4.00-5.00   sec   111 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   5.00-6.00   sec   111 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   6.00-7.00   sec   111 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   7.00-8.00   sec   112 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   8.00-9.00   sec   110 MBytes  0.11 GBytes/sec    0   4.15 MBytes
[  5]   9.00-10.00  sec   111 MBytes  0.11 GBytes/sec    0   4.15 MBytes
- - - - - - - - - - - - - - - - - - - - - - - - -
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-10.00  sec  1.09 GBytes  0.11 GBytes/sec    0             sender
[  5]   0.00-10.03  sec  1.09 GBytes  0.11 GBytes/sec                  receiver

iperf Done.
```
