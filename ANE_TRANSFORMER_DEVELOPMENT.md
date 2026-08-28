# ANE Transformer Development Record

## Project

Apple ML ANE Transformers — BNNS acceleration and Hugging Face/CoreML compatibility work.

## Development Environment

* Host: Apple Silicon Mac M4
* Shell environment: TermIDE
* Python environment: Python 3.14
* Repository: `ml-ane-transformers`
* Development branch: `bnns-accelerate-integration`
* Upstream: `apple/ml-ane-transformers`

## Repository State

The development branch is based on the upstream `main` lineage and contains the following development commits:

* `fa1a5d8` — Preserve ANE transformer implementation and tests
* `0e26807` — Normalize DistilBERT attention masks for ANE
* `09e6d27` — Add BNNS accelerator coverage
* `3fecbb5` — Add ANE CoreML transformer diagnostics and DistilBERT fixes

The latest development commit is currently pushed to:

`origin/bnns-accelerate-integration`

## Current Branch Delta

Compared with `origin/main`, the branch currently contains:

* BNNS graph accelerator integration
* BNNS accelerator tests
* DistilBERT attention-mask normalization
* CoreML input diagnostics
* traced-input inspection
* DistilBERT attention compatibility experiments
* attention-layout compatibility work
* attention return-value compatibility work
* result-index and result-tuple handling
* model-signature handling
* CoreML/DistilBERT test infrastructure
* preserved pre-fix DistilBERT test state

The complete branch delta is currently 17 files, approximately 1,827 additions and 2 deletions.

## Development Experiments

The following files represent diagnostic, compatibility, repair, or experimental work performed while investigating the DistilBERT → CoreML → ANE execution path:

* `diagnose_coreml_inputs.py`
* `inspect_traced_inputs.py`
* `fix_distilbert_attention.py`
* `fix_distilbert_attention_layout.py`
* `fix_distilbert_attention_return.py`
* `fix_distilbert_mask.py`
* `fix_distilbert_mask_variable.py`
* `fix_distilbert_result_index.py`
* `fix_distilbert_result_tuple.py`
* `fix_distilbert_signature.py`
* `repair_distilbert_attention.py`
* `test_distilbert_coreml.py`
* `ane_transformers/huggingface/test_distilbert.py.before_2input_coreml_fix`

These files are intentionally preserved as development history until their individual roles and final upstream suitability have been evaluated.

## Architectural Direction

The development path being investigated is:

Hugging Face Transformers
→ DistilBERT
→ ANE transformer implementation
→ BNNS acceleration
→ CoreML representation/execution
→ Apple Neural Engine / Apple Silicon

## Important Distinction

Experimental compatibility and diagnostic scripts should not automatically be interpreted as production-ready upstream changes.

Before upstream submission, each change should be classified as one of:

1. Required implementation
2. Required regression test
3. Diagnostic tooling
4. Temporary repair/experiment
5. Historical backup

## Preservation Policy

No experimental file should be deleted merely because it is not immediately suitable for upstream submission.

The development branch serves as the preserved research state. Cleanup or restructuring should occur in subsequent commits so the development history remains recoverable.

## Next Investigation

The next phase is to determine which experimental changes produce a reproducible, maintainable ANE/CoreML execution path for Hugging Face DistilBERT and which files should be consolidated into production implementation and regression tests.

