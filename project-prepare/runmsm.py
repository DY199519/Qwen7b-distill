import subprocess

# 要按顺序运行的 Python 文件（请按实际顺序填写文件名）
scripts_to_run = [
    'file1.6-Multsm.py',
    'file1.6-Multsm2.py',
    'file1.6-Multsm2fix.py',
    'file1.6-Multsm3.py',
    #mergepairwise.py，这个平时就不跑
    'file1.6-Multsm4-getscore-basic.py',
    'file1.6-Multsm4-getscore-context.py',
    'file1.6-Mulysm5-CountFormMsm.py',
    'file1.6-Multmm1.py',
    'file1.6-Multmm2-basictop234.py',
    'file1.6-Multmm3-getscore-claude.py',
    'file1.6-Multmm3-getscore-deepseek.py',
    'file1.6-Multmm3-getscore-gemini.py',
    'file1.6-Multmm3-getscore-qwen.py'
]

for script in scripts_to_run:
    print(f"\n🚀 正在运行：{script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ 已完成：{script}")
    except subprocess.CalledProcessError as e: 
        print(f"❌ 运行失败：{script}\n错误信息：{e}")
        break  # 如果某一步失败，终止后续脚本执行
