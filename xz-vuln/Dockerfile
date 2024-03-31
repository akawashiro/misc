FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y build-essential neovim openssh-server automake autopoint libtool bsdmainutils po4a doxygen libtool-bin
RUN mkdir /work
RUN mkdir /work/xz-install
COPY ./detect.sh /work/detect.sh
COPY ./xz-5.6.1 /work/xz-5.6.1
COPY ./xz-5.6.1 /work/DEBIAN
WORKDIR /work/xz-5.6.1
ENV RPM_ARCH x86_64
RUN ./autogen.sh
RUN ./configure
RUN make install -j
RUN mv ./src/liblzma/.libs/liblzma.so.5.6.1 /usr/lib/x86_64-linux-gnu/liblzma.so.5.2.5
WORKDIR /work
RUN ./detect.sh
