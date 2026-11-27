"""Produce complete candidate diagnostics and inference action plans, without callers."""
import ast
import copy
import csv
import importlib.metadata
import json
import shutil
from pathlib import Path

from build_routeB_swap_probe import split_prompt
from p0_nli_compare import apply, canon, group, key, summarize, text_pair, validate
from p0_nli_prepare import ROOT, OUT, STEP_B, read, sha

POLICIES = {'RelGate-0.50': 'relgate_mask', 'NLI-0.50': 'nli_mask_050', 'NLI-Matched': 'nli_mask_matched'}


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def main():
    rows = read(OUT / 'candidate_scores_actions.jsonl')
    counts = validate(rows)
    audit = json.loads((OUT / 'adaptation_audit.json').read_bytes())
    meta = json.loads((OUT / 'run_meta.json').read_bytes())
    assert meta['input_sha256'] == audit['input_sha256'] == sha(OUT / 'exploration_candidates.jsonl')
    assert meta['revision'] == meta['resolved_commit'] == audit['revision']
    assert meta['device'] == 'cpu' and meta['dtype'] == 'float32' and meta['threshold'] == .5
    inputs = read(OUT / 'exploration_candidates.jsonl')
    assert len(inputs) == len(rows)
    for original, scored in zip(inputs, rows):
        assert all(scored[k] == v for k, v in original.items())
    supplied = ROOT / 'artifact/relgate/reference/user_supplied_comparator.txt'
    def function_ast(source, name):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.dump(node, include_attributes=False)
    original_code = supplied.read_text(encoding='utf-8-sig')
    actual_code = (ROOT / 'scripts/p0_nli_compare.py').read_text(encoding='utf-8')
    for name in ('text_pair', 'canon', 'key'):
        assert function_ast(original_code, name) == function_ast(actual_code, name), name
    for r in inputs:
        changed = dict(r, changed=not r['changed'], is_target=not r['is_target'],
                       relgate_mask=not r['relgate_mask'], request_id='metadata-test',
                       view='metadata-test', candidate_index=r['candidate_index']+100)
        assert text_pair(r) == text_pair(changed) and key(r) == key(changed)
    fixture = [dict(request_id='tie', view='Clean', candidate_index=i, name=n,
                    description='Fixture.', parameters={}, changed=False, is_target=i == 0,
                    relgate_mask=i == 1) for i, n in enumerate(('alpha', 'beta'))]
    apply(fixture, {key(r): dict(p_contradiction=.5) for r in fixture})
    assert all(r['nli_mask_050'] for r in fixture)
    assert [r['nli_mask_matched'] for r in fixture] == [True, False]
    validation = dict(supplied_source_sha256=sha(supplied),
                      supplied_template_canon_key_ast_identical=True,
                      metadata_noninterference_checks=len(inputs),
                      threshold_includes_exact_050=True, matched_tie_break_by_candidate_index=True,
                      installed_dependencies_changed=False,
                      package_versions={n: importlib.metadata.version(n) for n in
                          ('torch', 'transformers', 'huggingface-hub', 'sentencepiece', 'scikit-learn', 'numpy', 'protobuf')})
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    metrics = summarize(rows)
    write_csv(OUT / 'candidate_metrics.csv', metrics)
    pool = {r['id']: r for r in read(STEP_B / 'unique_prompts.jsonl')}
    schedule = {(r['view'], r['id']): r for r in read(STEP_B / 'Qwen2.5-7B-Instruct.schedule.jsonl')
                if r['policy'] == 'Always-Mask' and r['view'] in counts}
    actions, request_metrics = [], []
    for (view, sid), rs in group(rows).items():
        rs.sort(key=lambda r: r['candidate_index'])
        assert sum(r['nli_mask_matched'] for r in rs) == sum(r['relgate_mask'] for r in rs)
        alias_cell = schedule[view, sid]
        aliases = [t['name'] for t in split_prompt(pool[alias_cell['prompt']])[1]]
        assert len(aliases) == len(rs)
        assert [alias_cell['inverse'][n] for n in aliases] == [r['name'] for r in rs]
        for policy, field in POLICIES.items():
            selected = [r['candidate_index'] for r in rs if r[field]]
            inverse = {aliases[i]: rs[i]['name'] for i in selected}
            visible = [aliases[i] if rs[i][field] else rs[i]['name'] for i in range(len(rs))]
            assert len(visible) == len(set(visible)) and len(inverse) == len(selected)
            assert [inverse.get(n, n) for n in visible] == [r['name'] for r in rs]
            actions.append(dict(request_id=sid, view=view, policy=policy, masked_indices=selected,
                                candidate_count=len(rs), masked_count=len(selected),
                                visible_names=visible, inverse=inverse))
            request_metrics.append(dict(request_id=sid, view=view, policy=policy,
                                        exact_construction_vector=int(all(r[field] == r['changed'] for r in rs)),
                                        proxy_candidate_accuracy=sum(r[field] == r['changed'] for r in rs)/len(rs),
                                        masked_count=len(selected), changed_count=sum(r['changed'] for r in rs)))
    (OUT / 'request_actions.jsonl').write_text(''.join(canon(r)+'\n' for r in actions), encoding='utf-8')
    write_csv(OUT / 'request_metrics.csv', request_metrics)
    lookup = {(r['view'], r['policy']): r for r in metrics}

    def pct(value):
        return 'N/A' if value is None else f'{value*100:.2f}%'

    lines = ['# P0-A: DeBERTa-NLI 与冻结 RelGate 的候选级对照', '',
             '## 实验范围', '',
             f"模型：`{meta['model']}`；固定提交：`{meta['resolved_commit']}`。",
             f"CPU FP32；Torch {meta['torch']}；Transformers {meta['transformers']}；线程 {meta['threads']}；batch={meta['batch_size']}；max_length={meta['max_length']}。",
             f"7 个冻结探索视图，{sum(counts.values()):,} 个请求视图单元，{len(rows):,} 个候选出现实例，{meta['unique_triples']:,} 个唯一三元组。",
             'RelGate-0.50 完全复用冻结动作；NLI-0.50 使用矛盾概率 ≥0.50；NLI-Matched 在每个请求内取矛盾概率最高的 k 个候选，k 等于该请求 RelGate 实际掩码数。分数并列时按候选下标排序。',
             '模型只接收名称、描述与参数 Schema。changed、is_target、请求 ID、查询和 RelGate 动作均不作为 NLI 输入。', '',
             '## 标签解释', '',
             '`changed` 表示相对同一候选槽位的 Clean 名称是否被构造性改名。它包括 Name/Target/Distractor 交换和 Verb-Flip 目标改名，**不等于人工确认的语义冲突真值**。以下“未改名掩码率”是构造代理误杀率；Clean 中的原始名称也不保证语义上完全准确。候选级结果不能直接推导调用器 Tool Acc 或端到端收益。', '',
             '## 未改名候选与金标名称保护', '',
             '| 视图 | 策略 | 未改名候选数 | 未改名掩码数 | 未改名掩码率 | 未改名金标数 | 金标掩码数 | 金标掩码率 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for view in ('Clean', 'Coarse-Desc', 'Verbose-Coarse', 'Target-Swapped', 'Distractor-Swapped'):
        for policy, field in POLICIES.items():
            r = lookup[view, field]
            lines.append(f"| {view} | {policy} | {r['unchanged']} | {r['fp']} | {pct(r['unchanged_mask_rate'])} | {r['target_unchanged_n']} | {r['target_false_masks']} | {pct(r['target_unchanged_mask_rate'])} |")
    lines.extend(['', '## 改名候选召回与选择', '',
                  '| 视图 | 策略 | 改名数 | 总掩码数 | 改名召回率 | 改名精确率 | 未改名掩码率 | 目标改名召回率 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|'])
    for view in ('Name-Swapped', 'Target-Swapped', 'Distractor-Swapped', 'MIXED', 'Verb-Flip-Target'):
        for policy, field in POLICIES.items():
            r = lookup[view, field]
            lines.append(f"| {view} | {policy} | {r['changed']} | {r['masked']} | {pct(r['recall'])} | {pct(r['precision'])} | {pct(r['unchanged_mask_rate'])} | {pct(r['target_changed_recall'])} |")
    lines.extend(['', '## 混合池同剂量比较', ''])
    rg = lookup['MIXED', 'relgate_mask']
    nl = lookup['MIXED', 'nli_mask_matched']
    lines.append(f"Target/Distractor 合计 364 个请求视图单元。RelGate 与 NLI-Matched 均掩码 {rg['masked']} 次；前者命中改名候选 {rg['tp']} 次，后者 {nl['tp']} 次。NLI-Matched 相对 RelGate 的代理精确率差为 {(nl['precision']-rg['precision'])*100:+.2f} 个百分点，代理召回率差为 {(nl['recall']-rg['recall'])*100:+.2f} 个百分点。差值按预定规则完整保留。")
    for policy in POLICIES:
        seq = [r for r in request_metrics if r['policy'] == policy and r['view'] in ('Target-Swapped', 'Distractor-Swapped')]
        lines.append(f"- {policy}：逐请求构造掩码向量完全匹配 {sum(r['exact_construction_vector'] for r in seq)}/{len(seq)}；逐请求候选代理准确率均值 {sum(r['proxy_candidate_accuracy'] for r in seq)/len(seq)*100:.2f}%。")
    lines.extend(['', '## 截断与耗时', '',
                  f"唯一候选截断 {meta['unique_truncated']}/{meta['unique_triples']}；按出现实例计 {meta['occurrence_truncated']}/{len(rows)}。仅截断前提（description + parameters），保留完整名称 hypothesis；阈值和模板未调整。",
                  f"本次 CPU 打分循环耗时 {meta['elapsed_cpu_seconds']:.1f} 秒；恢复的已有分数 {meta['resumed_unique_scores']} 条。耗时不含下载、加载及报告生成。", '',
                  '| 视图 | 候选出现数 | 截断数 | 截断率 |', '|---|---:|---:|---:|'])
    truncation = []
    for view in counts:
        seq = [r for r in rows if r['view'] == view]
        n = sum(r['truncated'] for r in seq)
        lines.append(f'| {view} | {len(seq)} | {n} | {n/len(seq)*100:.2f}% |')
        truncation.append(dict(view=view, candidates=len(seq), truncated=n, rate=n/len(seq)))
    write_csv(OUT / 'truncation_by_view.csv', truncation)
    lines.extend(['', '## 冻结与交付', '',
                  '全程本地 CPU；未连接 AutoDL，未训练或重新打分 RelGate，未调整 τ=0.50，未读取 334 条确认集，未执行调用器推理或 PA-Tool。',
                  '上传脚本的 text_pair、canon、key 函数 AST 与执行版一致。工程增强仅包括 CPU FP32 显式检查、断点保存、进度、模型配置与元数据、混合池汇总。',
                  '`candidate_scores_actions.jsonl` 保留三种策略全部候选动作；`request_actions.jsonl` 使用既有匿名名称与逆映射，为后续另行授权的推理准备动作。它不包含调用器输出。',
                  '`candidate_metrics.csv` 为完整候选指标；`request_metrics.csv` 为逐请求构造代理指标；`run_meta.json` 与 `model_config.json` 记录环境和模型。所有产物及执行脚本登记于 `SHA256SUMS.txt`。'])
    (OUT / 'NLI_RELGATE_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    code = OUT / 'code'
    code.mkdir(exist_ok=True)
    for name in ('p0_nli_compare.py', 'p0_nli_prepare.py', 'p0_nli_report.py'):
        shutil.copyfile(ROOT / 'scripts' / name, code / name)
    shutil.copyfile(supplied, code / 'user_supplied_comparator.txt')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    manifest = OUT / 'SHA256SUMS.txt'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files), encoding='ascii')
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha(OUT / name) == expected, name
    print('P0_REPORT_SHA256_VERIFIED', len(files), 'files;', len(actions), 'request action plans')


if __name__ == '__main__':
    main()
