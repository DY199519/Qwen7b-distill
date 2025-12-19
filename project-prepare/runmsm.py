import subprocess

# Python files to run in sequence (please fill in the filenames in the actual order)
scripts_to_run = [
    'file1.6-Multsm.py',
    'file1.6-Multsm2.py',
    'file1.6-Multsm2fix.py',
    'file1.6-Multsm3.py',
    #mergepairwise.py, this one is not run normally
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
    print(f"\n🚀 Running: {script}")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ Completed: {script}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Run failed: {script}\nError message: {e}")
        break  # If a step fails, terminate the execution of subsequent scripts
