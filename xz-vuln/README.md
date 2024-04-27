- <https://www.cisecurity.org/advisory/a-vulnerability-in-xz-utils-could-allow-for-remote-code-execution_2024-033>
- <https://www.openwall.com/lists/oss-security/2024/03/29/4>
- <https://ubuntu.com/security/CVE-2024-3094>
- <https://www.tenable.com/blog/frequently-asked-questions-cve-2024-3094-supply-chain-backdoor-in-xz-utils>

```bash
find . -type f -print | xargs grep '/dev/null' | grep "linux-gnu"
```

```
time env -i LANG=C /usr/sbin/sshd -h
```
