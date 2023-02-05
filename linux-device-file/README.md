# Linux Device File <!-- omit in toc -->
Linuxにおけるデバイスファイルとはデバイスをファイルという概念を通して扱えるようにしたものである。デバイスファイルは通常のファイルと同様に読み書きを行うことができる。しかし、実際には、その読み書きは例えばDMA等のデバイスとの情報のやり取りに変換される。デバイスファイルの例として、`/dev/nvme0` や `/dev/kvm` 等がある。

本稿では、デバイスファイルへの読み書きがどのようにデバイスへの制御に変換されるのかを述べる。ほとんどの内容は[詳解 Linuxカーネル 第3版](https://www.oreilly.co.jp/books/9784873113135/)の12章、13章に依る。参照しているLinux Kernelのソースコードのgitハッシュは`commit 830b3c68c1fb1e9176028d02ef86f3cf76aa2476 (v6.1)` である。

# 目次 <!-- omit in toc -->
- [デバイスドライバ](#デバイスドライバ)
  - [デバイスドライバを作ってみる](#デバイスドライバを作ってみる)
  - [`MyDeviceDriver.c` からわかること](#mydevicedriverc-からわかること)
- [ファイル](#ファイル)
  - [VFS(Virtual Filesytem Switch)](#vfsvirtual-filesytem-switch)
  - [inode](#inode)
  - [普通のファイルのinode](#普通のファイルのinode)
    - [デバイスファイルのinode](#デバイスファイルのinode)
- [デバイスドライバとファイルの接続](#デバイスドライバとファイルの接続)
  - [mknod](#mknod)
- [参考](#参考)

# デバイスドライバ
デバイスドライバとはカーネルルーチンの集合である。実際 [myDeviceDriver.c](./myDeviceDriver.c) を見ると単なる関数の集合であることがわかる。

## デバイスドライバを作ってみる
[組み込みLinuxデバイスドライバの作り方 (2)](https://qiita.com/iwatake2222/items/580ec7db2e88beeac3de)より引用。
```
> make
> sudo insmod MyDeviceModule.ko
> cat /proc/devices | grep 63
 63 MyDevice_NAME
> sudo mknod /dev/myDevice c 63 1
> file /dev/myDevice 
/dev/myDevice: character special (63/1)
> sudo chmod 666 /dev/myDevice
> echo "a" > /dev/myDevice
> sudo dmesg
[ 8969.738033] myDevice_init
[ 9160.327418] myDevice_open
[ 9160.327426] myDevice_write
[ 9160.327427] myDevice_write
[ 9160.327432] myDevice_close
```

## `MyDeviceDriver.c` からわかること
デバイスドライバは単なる関数の集合である。`open(2)`は`myDevice_open`を、`read(2)`は`myDevice_read`を呼び出す。また、`static` キーワードがついているのでリンク時にシンボルが外に公開されず、すべての関数は `s_myDevice_fops` 経由で呼び出される。

デバイスドライバはメジャー番号とマイナー番号を持ち、`MyDeviceDriver.c`の場合は`#define DRIVER_MAJOR 63`である。この番号はデバイスファイル作成時に `sudo mknod /dev/myDevice c 63 1` のように使われる。

# ファイル

## VFS(Virtual Filesytem Switch)
VFSとは標準的なUNIXファイルシステムのすべてのシステムコールを取り扱う、カーネルが提供するソフトウェアレイヤである。提供されているシステムコールとして`open(2)`、`close(2)`、`write(2)` 等がある。このレイヤがあるので、ユーザは`ext4`、`NFS`、`proc` などの全く異なるシステムを同じプログラムでで取り扱うことができる。例えば`cat(1)` は `cat /proc/self/maps` も `cat ./README.md`も可能だが、前者はメモリ割付状態を読み出しており、後者のファイル読み出しとはやっていることが本質的に異なる。

```
[@goshun](v6.1)~/linux
> cat /proc/self/maps | head
55b048a03000-55b048a05000 r--p 00000000 fd:01 10879784                   /usr/bin/cat
55b048a05000-55b048a09000 r-xp 00002000 fd:01 10879784                   /usr/bin/cat
55b048a09000-55b048a0b000 r--p 00006000 fd:01 10879784                   /usr/bin/cat
55b048a0b000-55b048a0c000 r--p 00007000 fd:01 10879784                   /usr/bin/cat
55b048a0c000-55b048a0d000 rw-p 00008000 fd:01 10879784                   /usr/bin/cat
55b04a820000-55b04a841000 rw-p 00000000 00:00 0                          [heap]
7fe5c1a11000-7fe5c208a000 r--p 00000000 fd:01 10881441                   /usr/lib/locale/locale-archive
7fe5c208a000-7fe5c208d000 rw-p 00000000 00:00 0 
7fe5c208d000-7fe5c20b5000 r--p 00000000 fd:01 10884115                   /usr/lib/x86_64-linux-gnu/libc.so.6
7fe5c20b5000-7fe5c224a000 r-xp 00028000 fd:01 10884115                   /usr/lib/x86_64-linux-gnu/libc.so.6
```

LinuxにおいてVFSはC言語を使った疑似オブジェクト志向で実装されている。つまり、関数ポインタを持つ構造体がオブジェクトとして使われている。

## inode
inodeオブジェクトとはとはVFSにおいて「普通のファイル」に対応するオブジェクトである。定義は [fs.h](https://github.com/akawashiro/linux/blob/830b3c68c1fb1e9176028d02ef86f3cf76aa2476/include/linux/fs.h#L588-L703) にある。他のオブジェクトとして、ファイルシステムそのものの情報を保持するスーパーブロックオブジェクト、オープンされているファイルとプロセスのやり取りの情報を保持するファイルオブジェクト、ディレクトリに関する情報を保持するdエントリオブジェクトがある。

```
struct inode {
	umode_t			i_mode;
	unsigned short		i_opflags;
	kuid_t			i_uid;
	kgid_t			i_gid;
	unsigned int		i_flags;
  ...
  union {
		struct pipe_inode_info	*i_pipe;
		struct cdev		*i_cdev;
		char			*i_link;
		unsigned		i_dir_seq;
	};
  ...
};
```

## 普通のファイルのinode
`stat 1`を使うとファイルのiノード情報を表示することができる。`struct inode` と対応していることがわかる。
```
[@goshun](master)~/misc/linux-device-file
> stat README.md 
  File: README.md
  Size: 20              Blocks: 8          IO Block: 4096   regular file
Device: fd01h/64769d    Inode: 49676330    Links: 1
Access: (0664/-rw-rw-r--)  Uid: ( 1000/   akira)   Gid: ( 1000/   akira)
Access: 2023-01-28 11:19:15.104727788 +0900
Modify: 2023-01-28 11:19:13.748734093 +0900
Change: 2023-01-28 11:19:13.748734093 +0900
 Birth: 2023-01-28 11:19:13.748734093 +0900
```

```
[@goshun](master)~/misc/linux-device-file
> stat --file-system README.md
  File: "README.md"
    ID: 968d19c9b6fe93c Namelen: 255     Type: ext2/ext3
Block size: 4096       Fundamental block size: 4096
Blocks: Total: 239511336  Free: 63899717   Available: 51714783
Inodes: Total: 60907520   Free: 52705995
```

### デバイスファイルのinode
先頭に`c` がついているとキャラクタデバイス、`b` がついているとブロックデバイス。
```
[@goshun]/dev
> ls -il /dev/nvme0*                                                         
201 crw------- 1 root root 240, 0  1月 29 19:02 /dev/nvme0
319 brw-rw---- 1 root disk 259, 0  1月 29 19:02 /dev/nvme0n1
320 brw-rw---- 1 root disk 259, 1  1月 29 19:02 /dev/nvme0n1p1
321 brw-rw---- 1 root disk 259, 2  1月 29 19:02 /dev/nvme0n1p2
322 brw-rw---- 1 root disk 259, 3  1月 29 19:02 /dev/nvme0n1p3
```

```
[@goshun](master)~/misc/linux-device-file
> stat /dev/nvme0n1
  File: /dev/nvme0n1
  Size: 0               Blocks: 0          IO Block: 4096   block special file
Device: 5h/5d   Inode: 319         Links: 1     Device type: 103,0
Access: (0660/brw-rw----)  Uid: (    0/    root)   Gid: (    6/    disk)
Access: 2023-01-28 10:03:26.964000726 +0900
Modify: 2023-01-28 10:03:26.960000726 +0900
Change: 2023-01-28 10:03:26.960000726 +0900
 Birth: -
```

# デバイスドライバとファイルの接続
## mknod
[mknod(2)](https://man7.org/linux/man-pages/man2/mknod.2.html)

# 参考
- [詳解 Linuxカーネル 第3版](https://www.oreilly.co.jp/books/9784873113135/)
- [https://github.com/torvalds/linux/tree/v6.1](https://github.com/torvalds/linux/tree/v6.1)
- [組み込みLinuxデバイスドライバの作り方](https://qiita.com/iwatake2222/items/1fdd2e0faaaa868a2db2)
