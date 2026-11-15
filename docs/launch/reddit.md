# Reddit draft

**A tiny offline preflight for PEFT LoRA adapters and local SafeTensors bases**

Before setting up a model loader, `lora-fitcheck check ./base ./adapter` tests whether a
saved adapter's tensor names, shapes and configured rank fit that local checkpoint. There
is an offline `demo` command and a JSON mode with separate statuses for mismatches and
checks that could not be completed. It opens only the required base headers for a sharded
model and does not read tensor payloads.

Version 0.1 is for standard dense 2D LoRA. Use a real load and inference test for runtime
behavior. Feedback on common export layouts or unclear findings would be useful.
