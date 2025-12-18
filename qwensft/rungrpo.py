#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Deployment script: Read local D:\project7\200.txt and upload to remote server for inference
"""

import paramiko
import time
import pathlib
import os

# ----------------------- Basic Configuration -----------------------
HOST = "connect.nma1.seetacloud.com"
PORT = 32831
USER = "root"
PWD = "52AiXsyd2yEe"
LOCAL_QUESTIONS = r"D:\project7\200.txt"  # Local question file
REMOTE_QUESTIONS = "/root/200.txt"        # Remote question file path
LOCAL_SCRIPT = __file__                   # Local script (for reference only)
REMOTE_SCRIPT = "/root/run_inference_grpo_remote2+1.py"  # Remote inference script path
REMOTE_LOG = "/root/vicuna_log-grpo2+1.out"       # Remote log path
MAX_RETRY = 5                             # SSH retry count
RETRY_GAP = 10                            # Retry interval in seconds

# ----------------------- SSH Helper Functions -----------------------
def connect_ssh():
    """SSH connection with automatic retry"""
    for i in range(1, MAX_RETRY + 1):
        try:
            print(f"🔌 SSH connection attempt {i}/{MAX_RETRY} ...")
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
            cli.get_transport().set_keepalive(30)
            print("✅ SSH connection successful!")
            return cli
        except Exception as e:
            print(f"⚠️ Connection failed: {e}")
            if i == MAX_RETRY:
                raise RuntimeError("❌ Failed after multiple reconnections, exiting") from e
            time.sleep(RETRY_GAP)

def main():
    """Main function"""
    # 1) Check local question file
    if not pathlib.Path(LOCAL_QUESTIONS).exists():
        raise FileNotFoundError(f"Local question file does not exist: {LOCAL_QUESTIONS}")
    
    # 2) SSH connection
    client = connect_ssh()
    
    # 3) Upload question file
    print("📤 Uploading question file ...")
    sftp = client.open_sftp()
    sftp.put(LOCAL_QUESTIONS, REMOTE_QUESTIONS)
    sftp.close()
    print("✅ Question file upload completed")
    
    # 4) Upload inference script
    print("📤 Uploading inference script ...")
    sftp = client.open_sftp()
    sftp.put("run_inference_grpo_remote2+1.py", REMOTE_SCRIPT)
    sftp.close()
    print("✅ Inference script upload completed")
    
    # 5) Check Swift CLI
    _, stdout, _ = client.exec_command('swift --help', timeout=5)
    if stdout.channel.recv_exit_status() == 0:
        print("✓ Swift command-line tool is available")
    else:
        print("⚠ Warning: Swift command not directly detected")
        print("Please ensure ms-swift is installed on the remote server (pip install ms-swift -U)")
        user_input = input("\nContinue running? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting program")
            client.close()
            return
    
    # 6) Check CUDA
    _, stdout, _ = client.exec_command('echo $CUDA_VISIBLE_DEVICES')
    cuda_device = stdout.read().decode().strip() or '0'
    print(f"✓ Using CUDA device: {cuda_device}")
    
    # 7) Run inference
    cmd = (
        f"nohup /root/miniconda3/bin/python {REMOTE_SCRIPT} {REMOTE_QUESTIONS} "
        f"> {REMOTE_LOG} 2>&1 &"
    )
    print("🚀 Sending background execution command ...")
    client.exec_command(cmd)
    print(f"✅ Inference script started in background, all outputs written to {REMOTE_LOG}")
    
    # 8) Poll logs
    print("⏳ Waiting for remote script output logs ...")
    time.sleep(5)
    _, stdout, _ = client.exec_command(f"head -n 10 {REMOTE_LOG} || echo '(Log not generated yet)'")
    print("------ Remote log preview ------")
    print(stdout.read().decode())
    
    client.close()
    print(f"\nInference task has started, please check {REMOTE_LOG} on the remote server for results")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted the program")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
