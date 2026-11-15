# Show HN draft

**Show HN: LoRAFit checks if a LoRA adapter fits before you load the model**

If you have a base snapshot and a separate PEFT adapter, can that adapter's A/B pairs fit the
exact weight names and dimensions in this checkpoint? The two artifacts can each look fine on
their own. LoRAFit reads their SafeTensors headers and the shard index, then reports which
tensor failed and why, with JSON and an exit code for CI.

Install from the repo and run `lorafit demo` to see a good match and a width mismatch.
The first release covers ordinary dense PEFT LoRA only. A pass does not identify the base
revision or predict runtime behavior. I'd like feedback on the findings you would need in a
model handoff, especially if your adapter format is outside this scope.
