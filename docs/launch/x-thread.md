# X thread draft

1. LoRA Fitcheck checks a saved PEFT adapter's names, shapes and rank against a local SafeTensors base before you load the model. Run `lora-fitcheck demo` for a match and a shape mismatch.
2. `lora-fitcheck check ./base ./adapter --format json` reads the index and relevant headers. Exits: 0 match, 1 mismatch, 2 check incomplete. No model runner or GPU needed.
3. The first version covers standard dense PEFT LoRA. A match does not verify the base revision or replace inference. If you've hit a confusing mismatch at handoff, I'd like to see a tiny reproducer.
