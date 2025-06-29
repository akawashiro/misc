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

```
$ iperf3 -c localhost --format G -bytes 1G
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
