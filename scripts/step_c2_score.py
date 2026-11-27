"""Run the frozen full-label likelihood implementation on C2 candidates."""
import json
import os
from pathlib import Path

from pareto_gpu import label_scores, sha


def main():
    root = Path("/root/autodl-tmp/Hammer")
    out = root / "research_outputs/step_c_confirm/c2"
    protocol = json.loads((out / "protocol.json").read_bytes())
    assert protocol["threshold"] == 0.5
    assert sha(out / "score_inputs.jsonl") == protocol["score_inputs_sha256"]
    adapter = root / "research_outputs/relgate_v2_final/RelGate-3Way-v2/adapter"
    assert sha(adapter / "adapter_model.safetensors") == "8403404d9518324bf16cb457dc25b795fb134309687a1c74313863215fa02259"
    os.environ.update(RELGATE_SCORE_ADAPTER=str(adapter), RELGATE_SCORE_INPUT=str(out / "score_inputs.jsonl"),
                      RELGATE_SCORE_OUTPUT=str(out / "scores"), PHASE25_LABELS_NO_EOS="1")
    label_scores("3way")


if __name__ == "__main__":
    main()
