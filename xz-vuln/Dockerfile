FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y build-essential neovim openssh-server automake autopoint libtool bsdmainutils po4a doxygen libtool-bin git time patchelf gawk bison texinfo gdb
RUN mkdir /work-xz
RUN mkdir /work-xz/xz-install
COPY ./detect.sh /work-xz/detect.sh
COPY ./xz-5.6.1 /work-xz/xz-5.6.1
COPY ./xz-5.6.1 /work-xz/DEBIAN

WORKDIR /work-xz/xz-5.6.1
ENV RPM_ARCH x86_64
RUN ./autogen.sh
RUN ./configure \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu
RUN make install -j
# RUN mv ./src/liblzma/.libs/liblzma.so.5.6.1 /usr/lib/x86_64-linux-gnu/liblzma.so.5.2.5

RUN mkdir -p /work-glibc/glibc-build
RUN mkdir -p /work-glibc/glibc-install
RUN git clone https://github.com/akawashiro/glibc.git /work-glibc/glibc
WORKDIR /work-glibc/glibc
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git fetch --all
RUN git checkout glibc-2.35-with-dl-audit-patch
WORKDIR /work-glibc/glibc-build
# RUN /work-glibc/glibc/configure --prefix /work-glibc/glibc-install
RUN /work-glibc/glibc/configure \
        --disable-sanity-checks \
        --prefix=/usr \
        --exec-prefix=/usr/lib/x86_64-linux-gnu
RUN make -j all
RUN make -j install || true

WORKDIR /work-xz
# RUN bash ./detect.sh
# RUN patchelf --set-interpreter /work-glibc/glibc-install/lib/ld-linux-x86-64.so.2 /usr/sbin/sshd
