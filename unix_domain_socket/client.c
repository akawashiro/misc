#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define UNIX_SOCKET_PATH "/tmp/test.unixsocket"
#define SEND_SIZE_MAX 256

// レスポンスの値
enum { OK, NG };

int unix_client();

int main(void) {
    int ret_code = 0;

    // クライアント処理を実行
    ret_code = unix_client();
    if (ret_code != 0) {
        printf("error\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int unix_client() {
    int ret_code = 0;
    char buf[SEND_SIZE_MAX];
    uint32_t buf_len = 0;

    int fd = -1;
    ssize_t size = 0;
    int flags =
        0;  // MSG_WAITALLとかMSG_NOSIGNALをよく使うけど今回はサンプルなのでフラグは無し

    int response = -1;  // レスポンス考えるの面倒だったので0:OK/1:NGで。

    // ソケットアドレス構造体
    struct sockaddr_un sun;
    memset(&sun, 0, sizeof(sun));

    // UNIXドメインのソケットを作成
    fd = socket(AF_LOCAL, SOCK_STREAM, 0);
    if (fd == -1) {
        printf("failed to create_socket(errno=%d:%s)\n", errno,
               strerror(errno));
        return -1;
    }

    // ソケットアドレス構造体に接続先(サーバー)を設定
    sun.sun_family = AF_LOCAL;
    strcpy(sun.sun_path, UNIX_SOCKET_PATH);

    // 上記設定を用いてサーバーに接続
    ret_code = connect(fd, (const struct sockaddr *)&sun, sizeof(sun));
    if (ret_code == -1) {
        printf("failed to create_socket(errno:%d, error_str:%s)\n", errno,
               strerror(errno));
        close(fd);
        return -1;
    }

    // 送信内容の作成(ここは自由)
    printf("please input what you want to send...\n>");
    if (fgets(buf, SEND_SIZE_MAX, stdin) == NULL) {
        printf("failed to fgets\n");
        close(fd);
        return -1;
    }
    buf_len = strlen(buf);
    printf("buf_len = %u\n", buf_len);

    // データサイズの送信(ヘッダーの構造とか考えなきゃだめだけど今回は省略。データサイズのみ送る)
    size = send(fd, &buf_len, sizeof(buf_len), flags);
    if (size < sizeof(buf_len)) {
        // partial sendパターンも今回はエラーとする
        printf("failed to send data size(errno:%d, error_str:%s)\n", errno,
               strerror(errno));
        close(fd);
        return -1;
    }

    // データ本体の送信
    size = send(fd, buf, buf_len, flags);
    if (size < buf_len) {
        // partial sendパターンも今回はエラーとする
        printf("failed to send data(errno:%d, error_str:%s)\n", errno,
               strerror(errno));
        close(fd);
        return -1;
    }

    // レスポンスの受信(今回は0:OK/1:NG)
    size = recv(fd, &response, sizeof(response), flags);
    if (size < sizeof(response)) {
        // partial recvパターンも今回はエラーとする
        printf("failed to recv response(errno:%d, error_str:%s)\n", errno,
               strerror(errno));
        close(fd);
        return -1;
    }

    printf("respose = %d\n", response);

    // ソケットを閉じる
    close(fd);

    return 0;
}

