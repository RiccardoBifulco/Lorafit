# Show HN draft

**Show HN: LoRA Fitcheck – Check a PEFT adapter against a local base before loading it**

If you have a base snapshot and a separate PEFT adapter, can that adapter's A/B pairs fit the
exact weight names and dimensions in this checkpoint? The two artifacts can each look fine on
their own. Fitcheck reads their SafeTensors headers and the shard index, then reports which
tensor failed and why, with JSON and an exit code for CI.

Install from the repo and run `lora-fitcheck demo` to see a good match and a width mismatch.
The first release covers ordinary dense PEFT LoRA only. A pass does not identify the base
revision or predict runtime behavior. I'd like feedback on the findings you would need in a
model handoff, especially if your adapter format is outside this scope.
