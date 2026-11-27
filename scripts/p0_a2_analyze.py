"""Score frozen P0-A2 Qwen predictions and paired exploratory contrasts."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from routeb_stage1 import remap
from score_routeB_probe import outcome
from step_b_analyze import keys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research_outputs/p0_a2'
STEP = ROOT / 'research_outputs/step_b_plugplay'
P0 = ROOT / 'research_outputs/p0_nli'
MODEL = 'Qwen2.5-7B-Instruct'
VIEWS = ('Clean', 'Coarse-Desc', 'Name-Swapped', 'Target-Swapped',
         'Distractor-Swapped', 'Verbose-Coarse', 'Verb-Flip-Target')
COUNTS = dict(zip(VIEWS, (197, 197, 197, 182, 182, 197, 72)))
POLICIES = ('Keep-Name', 'Always-Mask', 'Length-64', 'RelGate-0.50', 'NLI-0.50', 'NLI-Matched')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_csv(path, records):
    assert records
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def bootstrap(records):
    ids = sorted({r['id'] for r in records if r['view'] == 'Clean'})
    assert len(ids) == 197
    place = {sid: i for i, sid in enumerate(ids)}
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(ids), size=(20000, len(ids)))
    weights = np.zeros((20000, len(ids)), dtype=np.uint16)
    np.add.at(weights, (np.arange(20000)[:, None], draws), 1)
    values = np.zeros((len(ids), len(VIEWS), len(POLICIES), 2), dtype=np.float32)
    present = np.zeros((len(ids), len(VIEWS), len(POLICIES)), dtype=np.float32)
    for r in records:
        i, j, k = place[r['id']], VIEWS.index(r['view']), POLICIES.index(r['policy'])
        assert present[i, j, k] == 0
        present[i, j, k] = 1
        values[i, j, k] = r['tool_correct'], r['strict_correct']
    assert all(int(present[:, j, k].sum()) == COUNTS[v]
               for j, v in enumerate(VIEWS) for k in range(len(POLICIES)))
    num = (weights @ values.reshape(len(ids), -1)).reshape(20000, len(VIEWS), len(POLICIES), 2)
    den = (weights @ present.reshape(len(ids), -1)).reshape(20000, len(VIEWS), len(POLICIES))
    rates = np.divide(num, den[..., None], out=np.zeros_like(num), where=den[..., None] != 0)
    point = values.sum(0) / present.sum(0)[..., None]
    result = []
    for metric_i, metric in enumerate(('tool_acc', 'strict_match')):
        for nli in ('NLI-0.50', 'NLI-Matched'):
            a, b = POLICIES.index('RelGate-0.50'), POLICIES.index(nli)
            for view in (*VIEWS, 'Target+Distractor'):
                if view == 'Target+Distractor':
                    indices = [VIEWS.index('Target-Swapped'), VIEWS.index('Distractor-Swapped')]
                    delta = rates[:, indices, a, metric_i].mean(1) - rates[:, indices, b, metric_i].mean(1)
                    estimate = point[indices, a, metric_i].mean() - point[indices, b, metric_i].mean()
                    n = 364
                else:
                    j = VIEWS.index(view)
                    delta = rates[:, j, a, metric_i] - rates[:, j, b, metric_i]
                    estimate = point[j, a, metric_i] - point[j, b, metric_i]
                    n = COUNTS[view]
                low, high = np.quantile(delta, [.025, .975])
                result.append({'comparison': 'RelGate-0.50 minus ' + nli, 'metric': metric,
                               'view': view, 'n_view_cells': n, 'resampling_unit': 'original_request_id',
                               'delta_pp': 100 * float(estimate), 'ci_low_pp': 100 * float(low),
                               'ci_high_pp': 100 * float(high), 'draws': 20000})
    return result


def main():
    for folder in (OUT, P0, STEP):
        for line in (folder / 'SHA256SUMS.txt').read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            assert sha(folder / name) == digest, (folder, name)
    historical = json.loads((STEP / 'raw' / MODEL / 'full.report.json').read_bytes())
    inference = json.loads((OUT / 'inference.report.json').read_bytes())
    preflight = json.loads((OUT / 'preflight.json').read_bytes())
    assert inference['count'] == preflight['new_unique_prompts'] == 896
    assert inference['input_sha256'] == sha(OUT / 'new_prompts.jsonl')
    assert inference['output_sha256'] == sha(OUT / 'predictions.new.jsonl')
    assert inference['historical_report_sha256'] == sha(STEP / 'raw' / MODEL / 'full.report.json')
    for field in ('generation', 'backend', 'dtype', 'max_num_seqs', 'max_model_len', 'chat_template_sha256'):
        assert inference[field] == historical[field], field
    old_path = STEP / 'raw' / MODEL / 'predictions.jsonl'
    assert historical['output_sha256'] == sha(old_path)
    old = {r['id']: r for r in read(old_path)}
    new = {r['id']: r for r in read(OUT / 'predictions.new.jsonl')}
    assert len(old) == 4125 and len(new) == 896 and not (set(old) & set(new))
    all_preds = old | new
    gold = keys()
    schedule = read(OUT / 'schedule.jsonl')
    assert len(schedule) == 1224 * 6
    scored = []
    for cell in schedule:
        pred = all_preds[cell['prompt']]
        recovered = dict(pred, parsed_pred=remap(pred['parsed_pred'], cell['inverse']))
        tool, strict, hijack, invalid = outcome(recovered, gold[cell['id'], cell['view']])
        scored.append({'id': cell['id'], 'view': cell['view'], 'policy': cell['policy'],
                       'prompt': cell['prompt'], 'source': 'new' if cell['prompt'] in new else 'Step-B-cache',
                       'tool_correct': int(tool), 'strict_correct': int(strict), 'invalid': int(invalid),
                       'name_hijack': int(hijack) if cell['view'] in ('Name-Swapped', 'Target-Swapped') else None,
                       'masked_count': cell['masked_count'], 'candidate_count': cell['candidate_count']})
    assert len(scored) == len({(r['id'], r['view'], r['policy']) for r in scored}) == 7344
    for policy in POLICIES:
        assert Counter(r['view'] for r in scored if r['policy'] == policy) == COUNTS
    historical_metrics = {(r['policy'], r['view']): r for r in csv.DictReader(
        (STEP / 'eight_view_metrics.csv').open(encoding='utf-8')) if r['model'] == MODEL}
    metrics = []
    for policy in POLICIES:
        for view in VIEWS:
            rows = [r for r in scored if r['policy'] == policy and r['view'] == view]
            n = len(rows)
            entry = {'model': MODEL, 'policy': policy, 'view': view, 'n': n,
                     'tool_acc': sum(r['tool_correct'] for r in rows) / n,
                     'strict_match': sum(r['strict_correct'] for r in rows) / n,
                     'invalid_rate': sum(r['invalid'] for r in rows) / n,
                     'NHR_abs': sum(r['name_hijack'] for r in rows) / n if view in ('Name-Swapped', 'Target-Swapped') else None}
            if policy in POLICIES[:4]:
                ref = historical_metrics[policy, view]
                for field in ('tool_acc', 'strict_match', 'invalid_rate'):
                    assert abs(float(ref[field]) - entry[field]) < 1e-12, (policy, view, field)
            metrics.append(entry)
    write_csv(OUT / 'scored_requests.csv', scored)
    write_csv(OUT / 'seven_view_metrics.csv', metrics)
    contrasts = bootstrap(scored)
    write_csv(OUT / 'paired_bootstrap_20000.csv', contrasts)
    indexed = {(r['view'], r['id'], r['policy']): r for r in scored}
    actions = {(r['view'], r['request_id'], r['policy']): r for r in read(P0 / 'request_actions.jsonl')}
    gold_slot = {(r['view'], r['request_id']): r['candidate_index'] for r in read(P0 / 'exploration_candidates.jsonl') if r['is_target']}
    discordant = []
    groups = defaultdict(list)
    for view in ('Name-Swapped', 'Verb-Flip-Target'):
        for sid in sorted({r['id'] for r in scored if r['view'] == view}):
            gate = actions[view, sid, 'RelGate-0.50']
            for policy in ('NLI-0.50', 'NLI-Matched'):
                nli = actions[view, sid, policy]
                if gate['masked_indices'] == nli['masked_indices']:
                    continue
                g = gold_slot[view, sid]
                rg_ok = indexed[view, sid, 'RelGate-0.50']['tool_correct']
                nl_ok = indexed[view, sid, policy]['tool_correct']
                pattern = ('both_success' if rg_ok and nl_ok else 'both_fail' if not rg_ok and not nl_ok
                           else 'relgate_only' if rg_ok else 'nli_only')
                row = {'id': sid, 'view': view, 'nli_policy': policy,
                       'relgate_gold_masked': int(g in gate['masked_indices']),
                       'nli_gold_masked': int(g in nli['masked_indices']),
                       'relgate_tool_correct': rg_ok, 'nli_tool_correct': nl_ok,
                       'relgate_strict_correct': indexed[view, sid, 'RelGate-0.50']['strict_correct'],
                       'nli_strict_correct': indexed[view, sid, policy]['strict_correct'],
                       'outcome_pattern': pattern}
                discordant.append(row)
                groups[(view, policy, pattern, row['relgate_gold_masked'], row['nli_gold_masked'])].append(row)
    write_csv(OUT / 'discordant_requests.csv', discordant)
    write_csv(OUT / 'discordance_summary.csv', [
        {'view': v, 'nli_policy': p, 'outcome_pattern': outcome,
         'relgate_gold_masked': rg, 'nli_gold_masked': nl, 'requests': len(rs)}
        for (v, p, outcome, rg, nl), rs in sorted(groups.items())])
    contrast_lookup = {(r['comparison'], r['metric'], r['view']): r for r in contrasts}
    mixed = contrast_lookup['RelGate-0.50 minus NLI-Matched', 'tool_acc', 'Target+Distractor']
    name = contrast_lookup['RelGate-0.50 minus NLI-Matched', 'tool_acc', 'Name-Swapped']
    flip = contrast_lookup['RelGate-0.50 minus NLI-0.50', 'tool_acc', 'Verb-Flip-Target']
    lines = ['# P0-A2 冻结 Qwen 调用器对照', '',
             '7 个探索视图，1,224 个请求视图单元；各策略使用相同的 Prompt 模板、模型、解码与评分规则，实际工具名称随策略动作变化。',
             f"历史缓存命中 {preflight['cache_hits']} 个唯一提示，新增推理 {len(new)} 个；全部旧策略与 Step B 指标逐项一致。", '',
             f"新增推理报告：{inference['elapsed_seconds']:.1f} 秒（含模型加载），JSON 解析率 {inference['json_parse_rate']*100:.2f}%，峰值显存 {inference['peak_gpu_memory_gib_nvidia_smi']:.2f} GiB。",
             '## 主要观察', '',
             f"Target/Distractor 混合池中，RelGate 相对 NLI-Matched 的 Tool Acc 差为 {mixed['delta_pp']:+.2f} 点，95% CI [{mixed['ci_low_pp']:+.2f}, {mixed['ci_high_pp']:+.2f}]。这是等掩码数量的端到端正向结果。",
             f"Name-Swapped 上 NLI-Matched 的点估计更高：RelGate 减 NLI-Matched 为 {name['delta_pp']:+.2f} 点，95% CI [{name['ci_low_pp']:+.2f}, {name['ci_high_pp']:+.2f}]。",
             f"Verb-Flip-Target 上 NLI-0.50 的点估计更高：RelGate 减 NLI-0.50 为 {flip['delta_pp']:+.2f} 点，95% CI [{flip['ci_low_pp']:+.2f}, {flip['ci_high_pp']:+.2f}]。后两项 CI 跨零，均完整保留。", '',
             '| 视图 | 策略 | Tool Acc | Strict Match | Invalid |', '|---|---|---:|---:|---:|']
    for row in metrics:
        lines.append(f"| {row['view']} | {row['policy']} | {row['tool_acc']*100:.2f}% | {row['strict_match']*100:.2f}% | {row['invalid_rate']*100:.2f}% |")
    lines.extend(['', '## 联合配对 Bootstrap', '',
                  '20,000 次、按原始请求 ID 在七视图和策略间联合重抽；95% 百分位 CI。以下为 RelGate 减 NLI，单位百分点；未做多重比较校正。', '',
                  '| 指标 | 对照 | 视图 | 差值 | 95% CI |', '|---|---|---|---:|---:|'])
    for row in contrasts:
        lines.append(f"| {row['metric']} | {row['comparison']} | {row['view']} | {row['delta_pp']:+.2f} | [{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] |")
    lines.extend(['', '## 不一致请求观察', '',
                  '下表仅计两种门控掩码向量不同的请求；金标掩码与配对预测是观察性描述，不作因果归因。', '',
                  '| 视图 | NLI 策略 | 不一致请求 | RelGate 独对 | NLI 独对 | 双对 | 双错 |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    for view in ('Name-Swapped', 'Verb-Flip-Target'):
        for policy in ('NLI-0.50', 'NLI-Matched'):
            seq = [r for r in discordant if r['view'] == view and r['nli_policy'] == policy]
            counts = Counter(r['outcome_pattern'] for r in seq)
            lines.append(f"| {view} | {policy} | {len(seq)} | {counts['relgate_only']} | {counts['nli_only']} | {counts['both_success']} | {counts['both_fail']} |")
    lines.extend(['', '逐请求金标名称掩码状态、配对成功/失败见 `discordant_requests.csv`；交叉计数见 `discordance_summary.csv`。',
                  '截断敏感性复用 P0-A 的 `truncation_by_view.csv` 与原候选分数；本轮未重新打分。',
                  '本轮只使用冻结探索视图，未读取 334 条独立确认集；未训练、调阈值、改模板或扩展调用器。'])
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    archive = OUT / 'code'
    archive.mkdir(exist_ok=True)
    for name in ('p0_a2_prepare.py', 'p0_a2_restore_weights.py', 'p0_a2_caller.py',
                 'p0_a2_analyze.py', 'p0_a2_plot.py'):
        shutil.copyfile(ROOT / 'scripts' / name, archive / name)
    manifest = OUT / 'RESULT_SHA256SUMS.txt'
    files = sorted(x for x in OUT.rglob('*') if x.is_file() and x.name != 'RESULT_SHA256SUMS.txt')
    manifest.write_text(''.join(f'{sha(x)}  {x.relative_to(OUT).as_posix()}\n' for x in files), encoding='ascii')
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(OUT / name) == digest
    print('P0_A2_ANALYSIS_VERIFIED', len(files), len(scored), len(discordant), flush=True)


if __name__ == '__main__':
    main()
