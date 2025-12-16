#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
部署脚本：读取本地D:\project7\200.txt并上传到远程服务器运行推理
"""

import paramiko
import time
import pathlib
import os

# ----------------------- 基本配置 -----------------------
HOST = "connect.nma1.seetacloud.com"
PORT = 32831
USER = "root"
PWD = "52AiXsyd2yEe"
LOCAL_QUESTIONS = r"D:\project7\200.txt"  # 本地问题文件
REMOTE_QUESTIONS = "/root/200.txt"        # 远程问题文件路径
LOCAL_SCRIPT = __file__                   # 本地脚本（仅用于参考）
REMOTE_SCRIPT = "/root/run_inference_remote2+1.py"  # 远程推理脚本路径
REMOTE_LOG = "/root/vicuna_log-sft2+1.out"       # 远程日志路径
MAX_RETRY = 5                             # SSH重试次数
RETRY_GAP = 10                            # 重试间隔秒数

# ----------------------- SSH 帮助函数 -----------------------
def connect_ssh():
    """带自动重试的 SSH 连接"""
    for i in range(1, MAX_RETRY + 1):
        try:
            print(f"🔌 SSH 连接尝试第 {i}/{MAX_RETRY} 次 …")
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
            cli.get_transport().set_keepalive(30)
            print("✅ SSH 连接成功！")
            return cli
        except Exception as e:
            print(f"⚠️ 连接失败: {e}")
            if i == MAX_RETRY:
                raise RuntimeError("❌ 多次重连仍失败，退出") from e
            time.sleep(RETRY_GAP)

def main():
    """主函数"""
    # 1) 检查本地问题文件
    if not pathlib.Path(LOCAL_QUESTIONS).exists():
        raise FileNotFoundError(f"本地问题文件不存在: {LOCAL_QUESTIONS}")
    
    # 2) SSH 连接
    client = connect_ssh()
    
    # 3) 上传问题文件
    print("📤 正在上传问题文件 …")
    sftp = client.open_sftp()
    sftp.put(LOCAL_QUESTIONS, REMOTE_QUESTIONS)
    sftp.close()
    print("✅ 问题文件上传完成")
    
    # 4) 上传推理脚本
    print("📤 正在上传推理脚本 …")
    sftp = client.open_sftp()
    sftp.put("run_inference_remote2+1.py", REMOTE_SCRIPT)
    sftp.close()
    print("✅ 推理脚本上传完成")
    
    # 5) 检查Swift CLI
    _, stdout, _ = client.exec_command('swift --help', timeout=5)
    if stdout.channel.recv_exit_status() == 0:
        print("✓ Swift命令行工具可用")
    else:
        print("⚠ 警告: 无法直接检测到Swift命令")
        print("请确保远程服务器已安装 ms-swift (pip install ms-swift -U)")
        user_input = input("\n继续运行？(y/n): ")
        if user_input.lower() != 'y':
            print("退出程序")
            client.close()
            return
    
    # 6) 检查CUDA
    _, stdout, _ = client.exec_command('echo $CUDA_VISIBLE_DEVICES')
    cuda_device = stdout.read().decode().strip() or '0'
    print(f"✓ 使用CUDA设备: {cuda_device}")
    
    # 7) 运行推理
    cmd = (
        f"nohup /root/miniconda3/bin/python {REMOTE_SCRIPT} {REMOTE_QUESTIONS} "
        f"> {REMOTE_LOG} 2>&1 &"
    )
    print("🚀 下发后台执行命令 …")
    client.exec_command(cmd)
    print(f"✅ 推理脚本已在后台启动，所有输出写入 {REMOTE_LOG}")
    
    # 8) 轮询日志
    print("⏳ 等待远端脚本输出日志 …")
    time.sleep(5)
    _, stdout, _ = client.exec_command(f"head -n 10 {REMOTE_LOG} || echo '(日志尚未生成)'")
    print("------ 远端日志预览 ------")
    print(stdout.read().decode())
    
    client.close()
    print(f"\n推理任务已启动，请检查远程服务器的 {REMOTE_LOG} 获取结果")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)