# KrishiAI — Four-Month Roadmap

Status column reflects reality as of this scaffold (2026-09-06), not
aspiration. Update it as phases actually complete — see
`docs/project_state.md` for the live version of this.

## Month 1 — Data + Research Foundation

| Weeks | Work | Status |
|---|---|---|
| 1–2 | Understand YOLO26, collect datasets, extract archives, audit, document provenance | Tooling built and smoke-tested (`inspect_datasets.py`, `extract_datasets.py`); **no real datasets ingested yet** |
| 3–4 | Canonical taxonomy, annotation validation, dataset conversion, dedup, leakage prevention, baseline experiment | Tooling built (`validate_images.py`, `validate_labels.py`, `normalize_classes.py`, `detect_duplicates.py`, `build_dataset.py`); **blocked on real data** |

**Deliverable:** Validated dataset + reproducible preprocessing pipeline +
baseline YOLO26 model — **not yet reached.**

## Month 2 — Model Development
Train baseline, evaluate, error analysis, imbalance investigation, model-size
/ augmentation / resolution / class-balance experiments (Sections 33–38).
**Not started** — depends entirely on Month 1's dataset construction.

## Month 3 — Application + Deployment
FastAPI inference service, `ModelManager`, model versioning, frontend/mobile
interface, visualization, confidence handling, deployment prototype
(Sections 45–53). **Not started.**

## Month 4 — Validation + Research Output
Independent final evaluation, robustness/latency testing, ablations, final
documentation and presentation (Sections 62–66, 91–94). **Not started.**

## Immediate next action
Get real data into `data/raw/` (Basil, the Aloe Vera archive, Mint), then:
```bash
python scripts/extract_datasets.py     # if anything is archived
python scripts/inspect_datasets.py     # produces the real Step 1 audit
python scripts/normalize_classes.py    # surfaces raw class names to map
```
Then fill in `configs/dataset.yaml` and resolve `docs/decisions.md` D-003
before touching `scripts/build_dataset.py` or any training config.
