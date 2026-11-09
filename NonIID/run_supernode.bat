@echo off
cd /d C:\Users\user\Desktop\FL\FedProx
set PYTHONIOENCODING=utf-8
set VIRTUAL_ENV=C:\Users\user\Desktop\FL\quickstart-pytorch\.venv
set PATH=C:\Users\user\Desktop\FL\quickstart-pytorch\.venv\Scripts;%PATH%
flower-supernode --insecure --superlink %SUPERLINK_IP%:8000 --host 127.0.0.1 --node-config "partition-id=0 num-partitions=2" > supernode_desktop_noniid.log 2>&1
