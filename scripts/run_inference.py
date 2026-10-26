"""Local-only end-to-end PulsoXI inference command."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from games import GAME, require
from pulsoxi.atd_adapter import load_tactical_features
from pulsoxi.inference import infer_labels, infer_labels_v2, infer_labels_v3, infer_labels_v3b, infer_labels_v1_e2, infer_labels_v3b_e2, diagnostics
from pulsoxi.submission import write_submission
from starter.load_smart_tagging import load as load_tags

def main():
    p=argparse.ArgumentParser(description="Generate a local PulsoXI possession submission")
    p.add_argument("--data-dir",default=None)
    p.add_argument("--out",default="outputs/local_inference/submission_v1.csv")
    p.add_argument("--diagnostics",default=None)
    p.add_argument("--candidate",choices=("v1","v2","v3","v3b","v1e2","v3be2"),default="v1")
    a=p.parse_args()
    require(["atd_meta","atd_raw","smart_tagging"],GAME,a.data_dir,"tactical")
    features,_,_=load_tactical_features(GAME,a.data_dir)
    events=load_tags(GAME,a.data_dir)
    runner = {"v1": infer_labels, "v2": infer_labels_v2, "v3": infer_labels_v3, "v3b": infer_labels_v3b, "v1e2": infer_labels_v1_e2, "v3be2": infer_labels_v3b_e2}[a.candidate]
    labels,_,known=runner(features,smart_tagging_events=events)
    write_submission(a.out,labels)
    if a.diagnostics:
        report=diagnostics(labels,known); Path(a.diagnostics).parent.mkdir(parents=True,exist_ok=True)
        Path(a.diagnostics).write_text(json.dumps(report,indent=2)+chr(10),encoding="utf-8")
    print(f"Wrote {a.candidate} submission to {a.out}")
    return 0
if __name__=="__main__": raise SystemExit(main())