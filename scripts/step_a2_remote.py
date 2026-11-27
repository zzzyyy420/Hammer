import hashlib
import json
import shlex
import sys
from pathlib import Path
import paramiko

root = Path(__file__).resolve().parents[1]
remote = '/root/autodl-tmp/Hammer/research_outputs/step_a2_incremental'
password = sys.stdin.readline().rstrip('\r\n')
c = paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.cqa1.seetacloud.com', port=39317, username='root', password=password, timeout=25)
del password

def run(command):
    _, out, err = c.exec_command(command)
    stdout, stderr = out.read().decode(errors='replace'), err.read().decode(errors='replace')
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError(f'{code}: {stderr} {stdout}')
    return stdout

if sys.argv[1] == 'launch_scores':
    local = root / 'research_outputs/step_a2_incremental/score_inputs.jsonl'
    run('mkdir -p ' + shlex.quote(remote))
    sftp = c.open_sftp()
    sftp.put(str(local), remote + '/score_inputs.jsonl')
    sftp.close()
    expected = hashlib.sha256(local.read_bytes()).hexdigest()
    assert run('sha256sum ' + shlex.quote(remote + '/score_inputs.jsonl')).split()[0] == expected
    command = ('source /root/autodl-tmp/hammer_setup/activate.sh; '
               'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PHASE25_LABELS_NO_EOS=1; '
               'export RELGATE_SCORE_ADAPTER=/root/autodl-tmp/Hammer/research_outputs/relgate_v2_final/RelGate-3Way-v2/adapter; '
               f'export RELGATE_SCORE_INPUT={remote}/score_inputs.jsonl RELGATE_SCORE_OUTPUT={remote}/scores; '
               '/root/autodl-tmp/envs/hammer/bin/python -u /root/autodl-tmp/Hammer/research_outputs/relgate_pareto/gpu.py scores 3way')
    print(run('nohup bash -lc ' + shlex.quote(command) + ' > ' + shlex.quote(remote+'/scores.log') + ' 2>&1 < /dev/null & echo $!'))
elif sys.argv[1] == 'status':
    print(run('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv; tail -n 12 ' + shlex.quote(remote+'/scores.log')))
elif sys.argv[1] == 'launch_queue':
    target = '/root/autodl-tmp/Hammer/research_outputs/relgate_pareto/run_step_a2.sh'
    local = root / 'scripts/run_step_a2.sh'
    sftp = c.open_sftp()
    sftp.put(str(local), target)
    sftp.close()
    assert run('sha256sum ' + shlex.quote(target)).split()[0] == hashlib.sha256(local.read_bytes()).hexdigest()
    print(run('nohup bash '+shlex.quote(target)+' > /root/autodl-tmp/Hammer/research_outputs/relgate_pareto/step_a2.log 2>&1 < /dev/null & echo $!'))
elif sys.argv[1] == 'fetch':
    sftp = c.open_sftp()
    local = root / 'research_outputs/step_a2_incremental'
    for folder, data, report in [('scores', '3way.jsonl', '3way.done.json'), ('', 'greedy_labels.jsonl', 'greedy.done.json')]:
        target = local / folder
        target.mkdir(parents=True, exist_ok=True)
        prefix = remote + ('/' + folder if folder else '') + '/'
        for name in (report, data):
            sftp.get(prefix + name, str(target / name))
        meta = json.loads((target / report).read_text(encoding='utf-8'))
        assert hashlib.sha256((target / data).read_bytes()).hexdigest() == meta['sha256']
        assert len((target / data).read_bytes().splitlines()) == meta['count']
        print('verified', folder, data, meta['count'])
    sftp.close()
elif sys.argv[1] == 'launch_greedy':
    sftp = c.open_sftp()
    local = root / 'scripts/step_a2_greedy.py'
    sftp.put(str(local), remote + '/greedy.py')
    sftp.close()
    assert run('sha256sum ' + shlex.quote(remote+'/greedy.py')).split()[0] == hashlib.sha256(local.read_bytes()).hexdigest()
    command = ('source /root/autodl-tmp/hammer_setup/activate.sh; '
               'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false; '
               '/root/autodl-tmp/Hammer/research_outputs/phase1_gpu_v1/evalenv/bin/python -u ' + remote + '/greedy.py')
    print(run('nohup bash -lc ' + shlex.quote('while [ ! -f '+remote+'/scores/3way.done.json ]; do sleep 5; done; '+command) + ' > ' + shlex.quote(remote+'/greedy.log') + ' 2>&1 < /dev/null & echo $!'))
c.close()
