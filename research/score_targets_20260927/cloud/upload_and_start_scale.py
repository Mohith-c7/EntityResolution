"""Transfer completed scale-pilot folds to the isolated VM and start trials.

This does not access the submission VM or configure shutdown. The data producer
publishes parquet files atomically; the manifest is published last. Training
verifies every transferred parquet/truth hash before fitting.
"""
import json
from pathlib import Path
import subprocess
import time


def main():
    data = Path('models/cpu_scale_stage2_20260927_e70007')
    host = 'azureuser@20.205.231.10'
    key = str(Path.home()/'.ssh/er-expcpu-0927-e70007')
    socket = '/tmp/er-expcpu-0927-e70007.sock'
    remote = '/home/azureuser/er-experiments/e70007'
    common = ['-i', key, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
              '-o', 'ConnectTimeout=15', '-o', 'ControlPath='+socket]
    files = ['early_stop.parquet', 'early_stop_truth.json', 'selection.parquet',
             'selection_truth.json', 'train.parquet', 'train_truth.json', 'manifest.json']
    deadline = time.monotonic()+5400
    for name in files:
        path = data/name
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Data preparation did not finish; no training started')
            time.sleep(5)
        destination = host+':'+remote+'/entity/'+str(path)
        print(json.dumps({'stage':'upload_started','file':name,'bytes':path.stat().st_size}),flush=True)
        subprocess.run(['scp', *common, str(path), destination], check=True)
        print(json.dumps({'stage':'upload_complete','file':name}),flush=True)
    subprocess.run(['ssh', *common, host,
        'test -f '+remote+'/run_cpu_scale_trials.sh'], check=True)
    command = ('nohup bash '+remote+'/run_cpu_scale_trials.sh > '+remote+
               '/logs/scale_runner.log 2>&1 < /dev/null & echo $!')
    pid = subprocess.check_output(['ssh', *common, host, command],text=True).strip()
    report = {'status':'scale_runner_launched','runner_pid':pid,'host':host,
              'data':str(data),'seeds':[70007,70008,70009],
              'submission_vm_touched':False,'auto_shutdown_configured':False}
    Path(__file__).with_name('cpu_scale_launch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
