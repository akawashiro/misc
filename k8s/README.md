仮想マシンの作成
```bash
vagrant destroy -f
vagrant up
vagrant halt
```

kubectlのインストール
```
curl -LO "https://dl.k8s.io/release/$(curl -LS https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod  u+x ./kubectl
mv kubectl ~/.local/bin
```

kubeadm のインストール
```
ansible-playbook kubeadam-install.yaml -i hosts --ask-become-pass -vvvv
```
