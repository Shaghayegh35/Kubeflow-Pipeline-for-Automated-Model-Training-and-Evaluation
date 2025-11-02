import os, subprocess, sys

def test_compile_pipeline():
    assert os.path.exists('pipeline.py')
    res = subprocess.run([sys.executable, 'pipeline.py', '--compile'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert os.path.exists('pipeline.json')
