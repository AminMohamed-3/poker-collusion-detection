set -e
/venv/main/bin/pip install -q 'huggingface-hub==1.31.0' 'tqdm==4.70.1' 'filelock==3.32.3'
/venv/main/bin/pip list 2>/dev/null | grep -i -E 'huggingface-hub|tqdm|filelock'
supervisorctl start opencode_tabpfn_production
sleep 60
supervisorctl status opencode_tabpfn_production
echo '===LOG==='
tail -c 2000 /workspace/poker/codex/tabpfn3_production_20260913/production.log
