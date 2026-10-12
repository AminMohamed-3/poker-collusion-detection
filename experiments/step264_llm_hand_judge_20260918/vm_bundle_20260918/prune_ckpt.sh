while true; do
  cd /workspace/runs/r1 2>/dev/null || { sleep 60; continue; }
  for d in $(ls -d checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    n=${d#checkpoint-}
    if [ $((n % 20)) -eq 0 ] && [ ! -d keep/$d ]; then mkdir -p keep/$d; cp $d/adapter_model.safetensors $d/adapter_config.json keep/$d/ 2>/dev/null; fi
  done
  for d in $(ls -d checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -2); do rm -rf $d; done
  sleep 120
done
