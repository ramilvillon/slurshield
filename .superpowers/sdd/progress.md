# ngword SDD progress
Task 1: complete (commits 7f67318..cc2354f, review clean)
Task 2: complete (commits cc2354f..16bcbc4, review clean)
Task 4: complete (commits 16bcbc4..12ac259, review clean)
Task 5: REVIEW findings (fix in progress)
  - IMPORTANT: ToxiCN loader used wrong column (toxic_type/string keys); real signal is `expression` int 0-3 -> maps to clean/explicit/implicit/action. Touches labels.py + test_labels.py + dataprep.py + SOURCES.md.
  - MINOR: mginoben validation split (~2.78k tl rows) unused.
  - MINOR: stratified_split empty-train for tiny (lang,label) buckets (n<... ) — guard.
  - MINOR(defer to final): llmjp.csv label string values under-documented in SOURCES.md.
Task 5: complete (commits 12ac259..9b9a2b6, review clean after 1 fix cycle: ToxiCN expression mapping corrected)
Task 8: complete (commits 9b9a2b6..c73a484, review clean)
BOUNDARY: Tasks 3, 6, 7 blocked on heavy-dep install (torch/transformers/optimum ~GB) + manual COLD/ToxiCN downloads + overnight training. Deferred to user.
Task 3: complete (commit 3a7655c, mmbert-small confirmed on MPS, loss=2.2508)
Task 6: complete (commits f9b4e87..b2deacb, review clean after 1 fix: report() returns per-class F1)
Task 7: complete (commit 36d7661, int8 ONNX export of mmbert-small proven end-to-end on throwaway model)
FINAL whole-branch review (opus): READY TO MERGE. All invariants hold, 25 passed/2 skipped.
Tracked post-merge follow-ups (non-blocking):
  - Adversarial eval hook not wired (spec §6/§11); needs hand-collected obfuscation data.
  - ToxiCN toxic-first collapse in SOURCES.md prose only, not in _load_raw_csv (rebuild footgun).
REMAINING (user-run): overnight `python -m ngword.train` -> `python -m ngword.export` -> `python -m ngword.evaluate_model`.
Post-training FIX: names.py length-tiered matching (was over-blocking Alex/Phoenix/Model/etc via short-substring matches). 31 tests pass.
