# Discovery and selection

Review date: 2026-11-15. The scores below are engineering judgments, not survey results.
Frequency describes where a problem recurs in the workflow; these sources do not establish
population rates. Independent links are separate reports, experiments, or maintainer responses
that document a concrete failure or workaround.

| Candidate and concrete problem | Who encounters it and when | Evidence | Workaround and remaining friction |
| --- | --- | --- | --- |
| **LoRA fit:** a saved A/B pair names a missing base tensor or cannot form an update of the expected shape. | PEFT users at adapter handoff, merge, or evaluation against a different base snapshot. | [AutoTrain merge size mismatch](https://github.com/huggingface/autotrain-advanced/issues/790); [vLLM prefix mismatch reported by a serving user](https://github.com/vllm-project/vllm/issues/34186); [llama.cpp user's conversion failure on extra adapter weights](https://github.com/ggml-org/llama.cpp/issues/11554). | Load and inspect errors, or write a state-dict script. Loaders check their own runtime, but diagnosis comes after setup; a name mismatch can also leave an adapter ineffective. The latter two reports illustrate adjacent runtime behavior, not bugs this tool claims to reproduce or fix. |
| **Frozen weights:** a finished run changed no intended weights or changed tensors that were meant to stay frozen. | Fine-tuners after changing optimizer, freeze, or merge settings. | [PEFT parameter movement check](https://huggingface.co/docs/peft/en/developer_guides/troubleshooting); [PyTorch user checking unmodified weights](https://discuss.pytorch.org/t/model-weights-apparently-not-being-modified/200997); [Microsoft LoRA user's manual comparison](https://github.com/microsoft/LoRA/issues/190). | Manually compare checkpoints or use [Weightscope](https://pypi.org/project/weightscope/) or [wvc](https://pypi.org/project/wvc/). The latter tools already automate tensor-level comparison; no clear gap. |
| **Sharded exports:** a copied checkpoint has a partial or missing shard and fails only at load time. | Researchers transferring large exports between workstations, nodes, or registries. | [Comfy Desktop partial download](https://github.com/Comfy-Org/Comfy-Desktop/issues/1322); [Diffusers missing or corrupt upload objects](https://github.com/huggingface/diffusers/issues/6285); [oMLX's missing-weight-shard workaround](https://github.com/jundot/omlx/blob/main/omlx/model_discovery.py). | Checksums require known-good digests; loader checks require an attempted load. [Weightlint](https://github.com/thisbejim/weightlint) already provides offline bundle preflight. |
| **Paired evals:** aggregate scores on the same examples hide case flips and paired uncertainty. | Model evaluators comparing two runs of the same benchmark. | [EleutherAI comparator request](https://github.com/EleutherAI/lm-evaluation-harness/issues/3831); [NVIDIA's paired comparison workflow](https://docs.nvidia.com/nemo/evaluator/nightly/tutorials/compare); [evalci reanalysis and method](https://arxiv.org/abs/2607.04429). | Compare aggregates or write paired bootstraps. [evalci](https://pypi.org/project/evalci/) and NVIDIA's comparator already cover the practical workflow. |
| **Prompt drift:** equivalent task instructions yield different scores after formatting or few-shot ordering changes. | Benchmark maintainers whenever they change templates or migrate evaluators. | [Sclar et al.'s experiments](https://arxiv.org/abs/2310.11324); [Hugging Face few-shot ordering experiments](https://github.com/huggingface/blog/blob/main/evaluation-structured-outputs.md); [CDLM's recorded prompt and document hashes](https://github.com/zhangshuibai/CDLM/tree/main/evaluation/codegen). | Diff prompts by hand, pin configuration, or sample formats. [EvalRepro](https://github.com/seva9523/EvalRepro) already fingerprints contracts and coverage; [FormatSpread](https://arxiv.org/abs/2310.11324) addresses sensitivity. |
| **Split overlap:** duplicate or related examples appear in both training and evaluation splits. | Curators when resplitting, joining, or augmenting datasets. | [Lee et al.'s train/validation overlap study](https://aclanthology.org/2022.acl-long.577/); [Kapoor and Narayanan's leakage survey](https://doi.org/10.1016/j.patter.2023.100804); [DataEval's duplicate and temporal leakage checks](https://dataeval.readthedocs.io/en/latest/concepts/Leakage.html). | Group-aware splitting prevents known group collisions and exact hashes find copies; neither alone finds near copies. [split-leakage-audit](https://github.com/Yongjian-Wang6/split-leakage-audit) already covers the narrow local ID/group/fingerprint workflow. |
| **Transform cache:** a preprocessing stage is recomputed or reuses stale data because a cache key misses a relevant input. | Dataset pipeline authors when they change workers, source files, or hidden dependencies. | [Dataset multiprocessing cache report](https://github.com/huggingface/datasets/issues/3044); [DVC dependency-based invalidation](https://doc.dvc.org/command-reference/repro); [joblib's explicit disk caching and invalidation](https://joblib.readthedocs.io/en/latest/user_guide/memory.html). | Disable caching and redo expensive processing, or explicitly model dependencies. A generic checker cannot discover arbitrary dependencies hidden inside a transform. |
| **Resume data order:** a restarted run silently sees samples again or in a different order. | Long-running training jobs after interruption and checkpoint restore. | [Accelerate dataloader report](https://github.com/huggingface/accelerate/issues/2823); [ms-swift streaming replay report](https://github.com/modelscope/ms-swift/issues/10519); [TorchData stateful loader workaround](https://meta-pytorch.org/data/main/stateful_dataloader_tutorial.html). | Save loader state or log IDs around restart. State and step semantics vary by runtime and distributed sampler; reliably reproducing them is beyond this release. |
| **Resume metrics:** resumed logging duplicates steps or hides older history. | Researchers reconstructing curves after preemption and run resume. | [Megatron double-logging report](https://github.com/NVIDIA/Megatron-LM/issues/6237); [W&B fix for stale resume overwriting history](https://github.com/wandb/wandb/releases); [TensorBoard writer's explicit purge step](https://docs.pytorch.org/docs/main/utils/tensorboard.html). | Purge stale steps or reconcile exports. An offline repair cannot recover overwritten history or infer whether repeated steps represent retries or separate measurements. |
| **GPU memory triage:** checkpoint size alone does not explain an OOM during serving startup or warm-up. | Engineers changing model, batching, runtime, or device settings. | [PyTorch allocator diagnostic report](https://github.com/pytorch/pytorch/issues/173049); [vLLM multimodal warm-up OOM report](https://github.com/vllm-project/vllm/issues/28956); [NVIDIA NIM memory troubleshooting](https://docs.nvidia.com/nim/large-language-models/2.1.6/troubleshooting/memory.html). | Collect allocator snapshots and runtime logs. Reliable advice needs the allocator, kernel, and workload context; a single generic estimate would mislead. |
| **Missing paper artifacts:** the code, settings, or data needed to reproduce a result are absent or hard to locate. | Paper readers at each reproduction attempt. | [Olszewski et al.'s artifact study](https://www.cise.ufl.edu/~butler/pubs/ccs23-olszewski.pdf); [NeurIPS reproducibility challenge task](https://reproml.org/neurips2019/task/); [REFORMS consensus recommendations](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361/). | Follow artifact links and check submission checklists. Automated URL checking cannot establish that the retrieved artifact is complete or relevant. |

## Scoring

Gap means how much of the **exact** workflow remains unserved. A low gap is grounds for
rejection even if the aggregate is competitive. Scores are 1–5, with an unweighted total.

| Candidate | Pain | Breadth | Gap | Feasibility | Demo | Share | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LoRA fit** | 5 | 4 | 5 | 5 | 5 | 4 | **28** |
| Frozen weights | 4 | 4 | 1 | 4 | 5 | 4 | 22 |
| Sharded exports | 4 | 4 | 1 | 5 | 5 | 4 | 23 |
| Paired evals | 4 | 5 | 1 | 4 | 5 | 4 | 23 |
| Prompt drift | 4 | 4 | 1 | 4 | 4 | 4 | 21 |
| Split overlap | 5 | 5 | 1 | 4 | 5 | 4 | 24 |
| Transform cache | 3 | 4 | 3 | 2 | 3 | 2 | 17 |
| Resume data order | 5 | 4 | 3 | 2 | 3 | 4 | 21 |
| Resume metrics | 3 | 4 | 3 | 3 | 4 | 3 | 20 |
| GPU memory triage | 5 | 5 | 3 | 2 | 4 | 5 | 24 |
| Missing paper artifacts | 4 | 5 | 2 | 3 | 3 | 4 | 21 |

## Why LoRA Fitcheck

Separate base and adapter downloads make the handoff failure easy to isolate. The saved A/B
matrix shapes and the base's checkpoint keys provide enough evidence to detect several concrete
mistakes before installing a model runner. The check is a useful complement to loading the
adapter and testing a real output; supplying the wrong same-shaped base remains possible.

I searched GitHub, PyPI and npm for `PEFT adapter compatibility`, `LoRA preflight`,
`LoRA adapter validator`, `safetensors adapter shape` and `LoRA base mismatch`. This is a review
of discoverable tools and their documented interfaces, not proof that no other script exists.

| Closest option | What it does | Remaining gap for this workflow |
| --- | --- | --- |
| [PEFT](https://huggingface.co/docs/peft/main/developer_guides/checkpoint) | Loads adapters into real models and handles many adapter configurations. | Run PEFT as the runtime check. Its documented loading flow starts with a model object; the file-only check is earlier and needs neither PEFT nor a model runtime. |
| [PEFT Doctor](https://pypi.org/project/peft-doctor/) | Checks training recipes and `LoraConfig` fields, including saved adapter config. | Configuration checks do not establish that the serialized A/B factors fit the exact tensors in a separate checkpoint. |
| [AdapterSentry](https://pypi.org/project/adaptersentry/) | Checks structural details and analyzes weights within one adapter. | Its documented scan takes one adapter, not the local base's shard index and target headers. |
| [LoRAForge](https://lishenghui.github.io/loraforge/peft.html) | Reads PEFT adapter metadata, verifies its own manifests, compresses adapters. | The documented inspection concerns the adapter, rather than checking its targets against a separate base checkpoint. |
| [Weightlint](https://github.com/thisbejim/weightlint) | Checks a local checkpoint bundle for malformed headers, missing shards and related issues. | Bundle integrity and fit between two separately valid bundles are different checks. |
| [ckptkit](https://github.com/stef41/ckptkit) / [mergekit](https://github.com/arcee-ai/mergekit) | Inspect/merge checkpoints or derive adapters as documented. | Useful if weights must be manipulated; no documented header-only, two-artifact PEFT fit verdict with a CI exit status was found. |
| [vLLM](https://docs.vllm.ai/en/latest/features/lora/) / [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.cpp) | Validate aspects of adapters at load or conversion time for their own runtimes. | Runtime support is a separate question. This release checks local PEFT dense tensor names and shapes without starting a runtime or converting files. |

The gap is concrete: two paths in, exact mismatch locations and a CI verdict out, with no
runtime dependencies or weight-payload reads. The [benchmark](../benchmarks/) and
[exporter round-trip](../examples/) substantiate those design claims.
