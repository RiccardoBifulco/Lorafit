"""Exercise the tool on genuine Transformers and PEFT exports of tiny random models."""

import subprocess
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    torch = import_module("torch")
    transformers = import_module("transformers")
    peft = import_module("peft")
    torch.manual_seed(73)

    def config(width: int) -> object:
        return transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=width,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

    with TemporaryDirectory(prefix="fitcheck-peft-") as tmp:
        root = Path(tmp)
        base, wrong, adapter = (root / name for name in ("base", "wrong-base", "adapter"))
        model = transformers.LlamaForCausalLM(config(32))
        model.save_pretrained(base, max_shard_size="4KB", safe_serialization=True)
        lora = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
            ),
        )
        lora.save_pretrained(adapter, safe_serialization=True)
        different = transformers.LlamaForCausalLM(config(48))
        different.save_pretrained(wrong, safe_serialization=True)
        for checkpoint, expected in ((base, 0), (wrong, 1)):
            result = subprocess.run(
                [sys.executable, "-m", "lora_fitcheck", "check", str(checkpoint), str(adapter)],
                check=False,
            )
            if result.returncode != expected:
                raise RuntimeError(
                    f"{checkpoint.name}: expected exit {expected}, got {result.returncode}"
                )


if __name__ == "__main__":
    main()
