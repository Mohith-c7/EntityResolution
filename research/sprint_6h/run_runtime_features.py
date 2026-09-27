"""Execute two independent, actual test-block feature timings."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
import json

B=Path('research/sprint_6h/release/neural_v2_actual_blocks')


def block(number):
    root=B/('block_'+str(number));start=time.perf_counter()
    subprocess.run([sys.executable,'scripts/reverse_competition.py','features',
        '--pairs',str(root/'pairs.parquet'),'--manifest',str(root/'manifest.json'),
        '--route-dir',str(B/'global_route'),'--index','research/sprint_6h/reverse_competition/test_s1_v2.sqlite',
        '--target-index-prefix','models/index_test','--output',str(root/'raw_reverse')],check=True)
    raw_seconds=time.perf_counter()-start;start=time.perf_counter()
    subprocess.run([sys.executable,'research/sprint_6h/sibling/train_reverse_adapter.py','join',
        '--pairs',str(root/'pairs.parquet'),'--manifest',str(root/'manifest.json'),
        '--reverse-features',str(root/'raw_reverse/features.parquet'),
        '--reverse-manifest',str(root/'raw_reverse/manifest.json'),
        '--route-plan',str(B/'global_route'),'--output',str(root/'combined_reverse')],check=True)
    (root/'feature_timing.json').write_text(json.dumps({'feature_seconds':raw_seconds,
        'join_seconds':time.perf_counter()-start,'execution':'two independent sequential query workers'})+'\n')


with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(block,[0,1]))
subprocess.run([sys.executable,'research/sprint_6h/prepare_runtime_neural_inputs.py',
    '--blocks',str(B),'--data-dir','dataset/test','--output','models/sprint_6h/neural/neural_inputs/runtime_actual'],check=True)
