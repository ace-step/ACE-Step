# Repository Guidelines

## Project Structure & Module Organization
ACE-Step is packaged under `acestep/`, which hosts the GUI launcher (`gui.py`), diffusion pipeline (`pipeline_ace_step.py`), and model utilities inside `models/` and `language_segmentation/`. Training helpers such as `trainer.py`, `trainer-api.py`, and dataset tooling live at the repo root alongside orchestration scripts (`infer.py`, `infer-api.py`, `convert2hf_dataset.py`). Default assets and demo prompts sit in `assets/` and `examples/`, while checkpoints and generated audio are staged in `checkpoints/` and `outputs/` (keep these out of commits). Reference LoRA configs and datasets live in `config/` and `data/`.

## Build, Test, and Development Commands
Create a Python 3.10–3.13 environment and install editable dependencies with `pip install -e .`. Launch the desktop/web UI via `acestep --checkpoint_path ~/.cache/ace-step/checkpoints --port 7865`. For scripted inference use `python infer.py --checkpoint_path ./checkpoints --output_path outputs/demo.wav`. LoRA fine-tuning entry point: `python trainer.py --checkpoint_dir ./checkpoints --dataset_path ./data/your_dataset_path --lora_config_path config/zh_rap_lora_config.json`. When exposing APIs locally, run `uvicorn trainer-api:app --host 0.0.0.0 --port 8000`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case for functions/modules, and UpperCamelCase for Lightning modules or FastAPI schemas. Keep imports grouped (stdlib, third-party, local) and prefer type hints on new public functions. Use `loguru.logger` for structured logging and keep CLI arguments wired through `click` or `argparse` as shown in `infer.py` and `trainer.py`. Update `requirements.txt` when a runtime dependency changes.

## Testing Guidelines
There is no automated suite yet; add targeted unit tests with `pytest` (preferred) under a new `tests/` package when introducing pure-Python logic. For pipeline changes, run a smoke test: `python infer.py --checkpoint_path ./checkpoints --output_path outputs/smoke.wav` and listen for artefacts. API or GUI updates should be validated by starting `acestep` and confirming prompt-to-audio flow. Document any large model weights needed for reviewers in the PR description.

## Maintenance Guardrails
Run `python scripts/check_deprecations.py --fail-on-warning` before sending reviews. The checker runs key CLIs with `PYTHONWARNINGS=default` and fails if project code emits new deprecation warnings; known third-party issues are ignored automatically. Capture remediation steps or upstream blockers in the PR description when the script reports findings.

## Commit & Pull Request Guidelines
Commits trend short and imperative (e.g., `Use sox backend for ogg`, `Update TRAIN_INSTRUCTION.md`). Keep related changes together and reference issues or PR numbers when relevant (`Fix decoder drift #123`). Before opening a PR, ensure dependency installations and inference commands succeed, describe reproducible steps, and attach audio samples or screenshots when behaviour changes.
