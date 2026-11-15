# X thread draft

1. LoRAFit checks a saved PEFT adapter's names, shapes and rank against a local SafeTensors base before you load the model. Run `lorafit demo` for a match and a shape mismatch.
2. `lorafit check ./base ./adapter --format json` reads the index and relevant headers. Exit `0` means match, `1` means mismatch and `2` means incomplete. The check runs without a model runner or GPU.
3. LoRAFit supports standard dense PEFT LoRA and reports structural fit. If you've hit a confusing mismatch at handoff, I'd like to see a tiny reproducer.
