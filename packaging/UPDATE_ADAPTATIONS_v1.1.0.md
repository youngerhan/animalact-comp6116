# v1.1.0 packaging adaptations

- New settings store paths relative to AnimalAct-SlowFast. Original settings are retained in original-settings-v1.1.0/.
- The bundle keeps its portable clips.read_json and additionally resolves operational members/selection paths in memory. Frozen experiment plans/provenance retain the historical paths as evidence.
- rebuild_from_checkpoints.py now honors ft256/inference_overrides.json so historical crop floor rounding is reproduced.
- swin_experiment.py rejects --resume when last.pth is absent. Release packages omit optimizer checkpoints; start a fresh output directory for retraining.
- Publication README text identifies the incremental release. Historical numbers, labels, selection decisions and checkpoint tensors are unchanged.
- final-file-sha256.json and run provenance describe the original training workspace, before portability adaptations. Release integrity is separately recorded in root SHA256SUMS.json and update-manifest.json.
- The baseline Windows runtime and wheelhouse already contain torchvision 0.22.1+cu128; no dependency change is required.
- Checkpoint/score portability checks are distinct from full retraining numerical identity. Cross-hardware identical retraining is not promised.
