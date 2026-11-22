"""Fast checks: data invariants and a tiny end-to-end train/eval run on CPU."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from indicpeft import data as D  # noqa: E402
from indicpeft.modeling import build, count_params  # noqa: E402
from indicpeft.train import RunConfig, run  # noqa: E402

TINY = "hf-internal-testing/tiny-random-BertForSequenceClassification"


def test_sib200_is_parallel_and_labelled():
    pa, ur = D.load_lang("pa"), D.load_lang("ur")
    assert (len(pa["train"]), len(pa["validation"]), len(pa["test"])) == (701, 99, 204)
    assert pa["test"]["index_id"] == ur["test"]["index_id"]       # same sentences
    assert pa["test"]["label"] == ur["test"]["label"]
    assert set(pa["train"]["category"]) == set(D.LABELS)


def test_stratified_subsample_keeps_all_labels():
    tr = D.load_lang("pa")["train"]
    sub = D.stratified_subsample(tr, 50, seed=0)
    assert len(sub) == 50 and set(sub["label"]) == set(range(len(D.LABELS)))
    assert sub["index_id"] == D.stratified_subsample(tr, 50, seed=0)["index_id"]  # deterministic


def test_lora_trains_far_fewer_params_than_full():
    full, _ = build(TINY, "full")
    lora, _ = build(TINY, "lora", lora_r=4)
    assert count_params(lora).trainable < count_params(full).trainable
    assert count_params(full).pct == 100.0


def test_end_to_end_run_and_resume(tmp_path):
    cfg = RunConfig(model=TINY, method="lora", train_lang="pa", eval_langs=["pa", "ur"],
                    train_size=32, epochs=1, experiment="test")
    rec = run(cfg, tmp_path)
    assert set(rec["test"]) == {"pa", "ur"}
    assert len(rec["test"]["ur"]["y_pred"]) == 204
    assert (tmp_path / f"{cfg.run_id()}.json").exists()
    again = run(cfg, tmp_path)                     # second call is skipped
    assert again["run_id"] == rec["run_id"]
    json.dumps(rec)
