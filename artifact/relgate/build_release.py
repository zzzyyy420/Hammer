"""Build and verify the public RelGate research artifact from frozen local outputs."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'artifact/relgate'
RESULTS = ROOT / 'results/relgate'
SCRIPT_DIR = ROOT / 'scripts'

ENTRYPOINTS = (
    'build_relgate_dataset.py', 'build_relgate_v2.py', 'relgate_train.py',
    'relgate_gpu.py', 'relgate_evaluate.py', 'relgate_v2_prepare_scores.py',
    'relgate_v2_greedy.py', 'relgate_v2_audit.py', 'relgate_v2_b2.py',
    'pareto_policies.py', 'pareto_experiment.py', 'routeb_stage1.py',
    'routeb_step2a.py', 'step_a_probes.py', 'step_a2_prepare_scores.py',
    'step_a2_greedy.py', 'step_a2_local_report.py', 'step_a2_remote.py',
    'step_b_prepare.py', 'step_b_caller_runner.py', 'step_b_vllm_runner.py',
    'step_b_analyze.py', 'step_b_plot.py', 'step_b_download.py',
    'step_b_validate.py', 'step_c_bfcl_adapter.py', 'step_c1_prepare.py',
    'step_c15_prepare.py', 'step_c2_freeze.py', 'step_c2_score.py',
    'step_c2_schedule.py', 'step_c2_caller.py', 'step_c2_analyze.py',
    'step_c2_microbench.py', 'step_c2_snapshot.py', 'step_c2_bench_prepare.py',
    'p0_nli_compare.py', 'p0_nli_prepare.py',
    'p0_nli_report.py', 'p0_a2_prepare.py', 'p0_a2_restore_weights.py',
    'p0_a2_caller.py', 'p0_a2_analyze.py', 'p0_a2_plot.py',
)
TESTS = ('test_pareto_policies.py', 'test_routeB_swap_cpu.py',
         'test_routeB_stage1_cpu.py', 'test_relgate_cpu.py')
TABLES = {
    'exploration/eight_view_metrics.csv': 'step_b_plugplay/eight_view_metrics.csv',
    'exploration/aggregate_metrics.csv': 'step_b_plugplay/aggregate_metrics.csv',
    'exploration/paired_bootstrap_20000.csv': 'step_b_plugplay/paired_bootstrap_20000.csv',
    'confirmation/heldout_seven_view_metrics.csv': 'step_c_confirm/c2/heldout_seven_view_metrics.csv',
    'confirmation/heldout_aggregate_metrics.csv': 'step_c_confirm/c2/heldout_aggregate_metrics.csv',
    'confirmation/bfcl_official_AST_metrics.csv': 'step_c_confirm/c2/bfcl_official_AST_metrics.csv',
    'confirmation/paired_bootstrap_20000.csv': 'step_c_confirm/c2/paired_bootstrap_20000.csv',
    'confirmation/cross_stage_shared_view_metrics.csv': 'step_c_confirm/c2/cross_stage_shared_view_metrics.csv',
    'nli/candidate_metrics.csv': 'p0_nli/candidate_metrics.csv',
    'nli/truncation_by_view.csv': 'p0_nli/truncation_by_view.csv',
    'nli/seven_view_metrics.csv': 'p0_a2/seven_view_metrics.csv',
    'nli/paired_bootstrap_20000.csv': 'p0_a2/paired_bootstrap_20000.csv',
    'nli/discordance_summary.csv': 'p0_a2/discordance_summary.csv',
    'nli/dual_pareto.svg': 'p0_a2/dual_pareto.svg',
    'nli/caller_comparison.md': 'p0_a2/REPORT.md',
}
WEIGHTS = {
    'relgate_3way_v2': 'relgate_v2_final/RelGate-3Way-v2/adapter',
    'hammer_repro_1p5b': 'phase1_gpu_v1/Pure-NameMask',
    'b2_1p5b': 'routeB_step2b_v1/B2/attempt1/adapter',
}
TEMPLATE = 'relgate_v1/data/external_gate_prompt_v1.txt'
REFERENCES = {
    'caller_model_provenance.json': 'step_b_plugplay/model_provenance.json',
    'exploration_analysis_protocol.json': 'step_b_plugplay/analysis_protocol.json',
    'confirmation_protocol.json': 'step_c_confirm/c2/protocol.json',
    'confirmation_analysis_protocol.json': 'step_c_confirm/c2/analysis_protocol.json',
    'relgate_training_verified.json': 'relgate_v2_final/RelGate-3Way-v2/verified.json',
    'relgate_training_audit.json': 'relgate_v2_final/audit_v2.json',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def sync(path, content, check):
    path.parent.mkdir(parents=True, exist_ok=True)
    if check:
        assert path.read_bytes() == content, path
    else:
        path.write_bytes(content)


def copy(source, target, check):
    assert source.is_file(), source
    target.parent.mkdir(parents=True, exist_ok=True)
    if check:
        assert sha(source) == sha(target), target
    else:
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)


def source_closure():
    local = {path.stem: path for path in SCRIPT_DIR.glob('*.py')}
    pending = [Path(name).stem for name in ENTRYPOINTS]
    chosen = set()
    for name in TESTS:
        tree = ast.parse((ROOT / 'tests' / name).read_text(encoding='utf-8'))
        pending.extend(imports(tree, local))
    while pending:
        name = pending.pop()
        if name in chosen:
            continue
        assert name in local, name
        chosen.add(name)
        tree = ast.parse(local[name].read_text(encoding='utf-8'))
        pending.extend(imports(tree, local))
    return sorted('scripts/' + name + '.py' for name in chosen) + sorted('tests/' + name for name in TESTS)


def imports(tree, local):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            name = node.module.split('.')[0]
            if name in local:
                found.append(name)
        elif isinstance(node, ast.Import):
            found.extend(alias.name.split('.')[0] for alias in node.names
                         if alias.name.split('.')[0] in local)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result_entries = []
    for published, relative in sorted(TABLES.items()):
        source = ROOT / 'research_outputs' / relative
        target = RESULTS / published
        copy(source, target, args.check)
        result_entries.append({'published': target.relative_to(ROOT).as_posix(),
                               'source': source.relative_to(ROOT).as_posix(),
                               'sha256': sha(source), 'bytes': source.stat().st_size})
    weight_entries = []
    for name, relative in sorted(WEIGHTS.items()):
        for file in ('adapter_model.safetensors', 'adapter_config.json'):
            source = ROOT / 'research_outputs' / relative / file
            target = RELEASE / 'weights' / name / file
            copy(source, target, args.check)
            weight_entries.append({'name': name, 'published': target.relative_to(ROOT).as_posix(),
                                   'source': source.relative_to(ROOT).as_posix(),
                                   'sha256': sha(source), 'bytes': source.stat().st_size})
    template_source = ROOT / 'research_outputs' / TEMPLATE
    template_target = RELEASE / 'templates/external_gate_prompt_v1.txt'
    copy(template_source, template_target, args.check)
    comparator_source = ROOT / 'research_outputs/p0_nli/code/user_supplied_comparator.txt'
    comparator_target = RELEASE / 'reference/user_supplied_comparator.txt'
    copy(comparator_source, comparator_target, args.check)
    p0_validation = json.loads((ROOT / 'research_outputs/p0_nli/validation.json').read_bytes())
    assert p0_validation['supplied_source_sha256'] == sha(comparator_source)
    protocol = json.loads((ROOT / 'research_outputs/step_c_confirm/c2/protocol.json').read_bytes())
    assert protocol['gate'] == 'RelGate-3Way-v2'
    assert protocol['gate_score'].endswith('CONTRADICTED >= 0.50')
    assert protocol['inputs_sha256']['research_outputs/' + TEMPLATE] == sha(template_source)
    for record in weight_entries:
        key = record['source']
        assert protocol['inputs_sha256'][key] == record['sha256'], key
    inputs = [{'published': template_target.relative_to(ROOT).as_posix(),
               'source': template_source.relative_to(ROOT).as_posix(),
               'sha256': sha(template_source), 'bytes': template_source.stat().st_size},
              {'published': comparator_target.relative_to(ROOT).as_posix(),
               'source': comparator_source.relative_to(ROOT).as_posix(),
               'sha256': sha(comparator_source), 'bytes': comparator_source.stat().st_size}]
    for published, relative in sorted(REFERENCES.items()):
        source = ROOT / 'research_outputs' / relative
        target = RELEASE / 'reference' / published
        copy(source, target, args.check)
        inputs.append({'published': target.relative_to(ROOT).as_posix(),
                       'source': source.relative_to(ROOT).as_posix(),
                       'sha256': sha(source), 'bytes': source.stat().st_size})
    source_paths = source_closure()
    source_entries = [{'path': name, 'sha256': sha(ROOT / name), 'bytes': (ROOT / name).stat().st_size}
                      for name in source_paths]
    release = {
        'protocol': 'frozen RelGate-3Way-v2, contradiction score >= 0.50',
        'source_files': source_entries,
        'template': inputs,
        'adapters': weight_entries,
        'published_results': result_entries,
        'external_bfcl_commit': '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8',
        'private_inputs_excluded': True,
        'raw_predictions_excluded': True,
    }
    sync(RELEASE / 'manifest.json', canonical(release), args.check)
    result_lines = ''.join(f"{row['sha256']}  {row['published']}\n" for row in result_entries)
    sync(RELEASE / 'RESULTS_SHA256SUMS.txt', result_lines.encode('ascii'), args.check)
    all_paths = sorted(set(source_paths + [
        'README.md', '.gitignore', '.gitattributes',
        'configs/hammer15b_train_v1/base_model_revision.json',
        'artifact/relgate/README.md', 'artifact/relgate/ENVIRONMENT.md',
        'artifact/relgate/DATA.md', 'artifact/relgate/build_release.py',
        'artifact/relgate/verify_public.py',
        'artifact/relgate/manifest.json', 'artifact/relgate/RESULTS_SHA256SUMS.txt',
        'artifact/relgate/release_paths.txt',
    ] + [row['published'] for row in result_entries + weight_entries + inputs]))
    sync(RELEASE / 'release_paths.txt', ''.join(path + '\n' for path in all_paths).encode('utf-8'), args.check)
    print(f'RELEASE_VERIFIED source={len(source_paths)} results={len(result_entries)} '
          f'adapter_files={len(weight_entries)} paths={len(all_paths)}')


if __name__ == '__main__':
    main()
