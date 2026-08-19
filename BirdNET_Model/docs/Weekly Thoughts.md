## What I did this week

- **Audited the inherited BirdNET benchmarks.** The "FP16 collapse" finding turned out to be a measurement artifact, not a real result.
  - Proof: the file labeled FP16 is actually a full FP32 model (zero float16 tensors), loaded through the wrong birdnetlib slot. Plus a location-filter confound and a broken memory metric.
  - Decision: stop trusting the inherited compression numbers; Qian's FP16 result (26 MB, 95.3%) is the one valid baseline.

- **Built a skeleton pack** (manifest + pack.py) against Daan's draft contract — on purpose, to surface unknowns before the real models exist.
  - Proposed a single unified output table with a `detection_type` discriminator, and a `role` field for weights.

- **Ran the contract and model questions past Daan and Qian.** Their answers reshaped the architecture:
  - HP is binary and feature-fed, not multi-class; AED is a gate we keep as diagnostic; BirdNET is dual-purpose, with a feature-engineering step feeding HP.

- **Locked the compression direction:** class pruning as the lead (training-free, configurable around a TBD target-species list); INT8 timeboxed to a per-channel test; FP16 proven viable.

## How I'm making decisions

- Verify inherited findings before building on them (the FP16 audit is the example).
- Design to the contract early to flush out unknowns while they're cheap to change.
- Keep designs configurable when inputs are still TBD (pruning around a species list we don't have yet).
- Separate confirmed facts from assumptions, and route each question to its owner.

## Up next

- Review the AED and HP repos to replace assumptions with real I/O after access to HP and AED repos.
- Reproduce the FP16 baseline on a clean ONNX harness — built to also capture system metrics (time, memory, model size) for Kyle's twin.
- Settle the output schema (unified table + diagnostic channel vs. multi-table) once repo fields are confirmed.
- Build the class-pruning mechanism, configurable, ready to drop in the target-species list.
