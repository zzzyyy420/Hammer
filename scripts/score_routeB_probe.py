"""Score complete greedy generations against the frozen view-specific keys."""
import csv
from pathlib import Path
import sys
from paired_did_eval import indexed, rows, score, call_parts

MODELS = ("Pure-NameMask", "Hammer-1.5b", "Pure-NoMask", "Qwen2-1.5B-Instruct")
VIEWS = ("Name-Swapped", "Partial-Swap", "Name-Masked")


def outcome(pred, key):
    invalid, selected, strict = score(pred, key, False)
    calls = pred["parsed_pred"]
    hijack = bool(not selected and pred["parse_ok"] and isinstance(calls, list)
                  and len(calls) == 1 and isinstance(calls[0], dict)
                  and call_parts(calls[0])[0] == key["original_gold"])
    return selected, strict, int(hijack), invalid


def main(folder):
    table = []
    for model in MODELS:
        for view in VIEWS:
            keys = indexed(rows(folder/"private"/(view+".jsonl")))
            preds = indexed(rows(folder/"predictions"/model/(view+".jsonl")))
            assert keys.keys() == preds.keys() and len(keys) == 200
            correct, strict, hijacks, invalid = map(sum, zip(*(outcome(preds[i], k) for i,k in keys.items())))
            n = len(keys)
            swapped = view != "Name-Masked"
            table.append(dict(model=model, view=view, n=n, correct=correct, tool_acc=correct/n,
                              tool_arg=strict/n, invalid=invalid, errors=n-correct,
                              hijacks=hijacks if swapped else None,
                              nhr_errors=hijacks/(n-correct) if swapped and n>correct else None,
                              nhr_all=hijacks/n if swapped else None))
    with (folder/"metrics.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=table[0])
        w.writeheader()
        w.writerows(table)
    report = ["| Model | Masked | Swapped | Drop (pp) | Hijacked / errors | Partial | Partial hijacked / errors |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        v = {r["view"]: r for r in table if r["model"] == model}
        masked, swapped, partial = (v[n] for n in ("Name-Masked", "Name-Swapped", "Partial-Swap"))
        drop = 100*(masked["correct"]-swapped["correct"])/masked["n"]
        report.append(f'| {model} | {masked["correct"]}/200 | {swapped["correct"]}/200 | {drop:.1f} | '
                      f'{swapped["hijacks"]}/{swapped["errors"]} | {partial["correct"]}/200 | '
                      f'{partial["hijacks"]}/{partial["errors"]} |')
    (folder/"probe_summary.md").write_text("\n".join(report)+"\n", encoding="utf8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
