"""
FLUX.2-dev image generation backend for miurba-generator.

This is the OPTIONAL image-generation backend used when
`image_generation.enabled` is true in the shared config. It loads
FLUX.2-dev 32B (NF4) distributed across two GPUs
(text_encoder on cuda:1, transformer+VAE on cuda:0) without CPU offload.

Heavy imports (torch, diffusers, transformers, huggingface_hub) are lazy so
this module can be imported in environments without a GPU or without those
deps when image generation is disabled.
"""

import json
import os
from typing import Any


class FluxImageGenerator:
    """Generate images with FLUX.2-dev 32B (NF4) on two GPUs."""

    def __init__(
        self,
        cache: str = "diffusers/FLUX.2-dev-bnb-4bit",
        size: int = 1024,
        steps: int = 28,
        guidance: float = 4.0,
        device0: str = "cuda:0",
        device1: str = "cuda:1",
    ):
        self.cache = cache
        self.size = size
        self.steps = steps
        self.guidance = guidance
        self.device0 = device0
        self.device1 = device1
        self.pipe: Any = None
        self._device0 = None

    def load(self):
        """Load the FLUX pipeline distributed across two GPUs."""
        import torch
        from diffusers import (
            Flux2Pipeline,
            Flux2Transformer2DModel,
            AutoencoderKLFlux2,
        )
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import Mistral3ForConditionalGeneration, AutoProcessor
        from huggingface_hub import hf_hub_download

        sched_path = hf_hub_download(self.cache, "scheduler/scheduler_config.json")
        with open(sched_path) as f:
            scheduler_config = json.load(f)

        device0 = torch.device(self.device0)
        device1 = torch.device(self.device1)

        print("Cargando componentes FLUX.2...")

        # Text encoder -> GPU1 (permanentemente, ~14GB)
        print(f"  [GPU1] Text encoder (Mistral3 NF4)...")
        t0 = __import__("time").time()
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            self.cache, subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map={"": self.device1},
        )
        print(f"  {__import__('time').time()-t0:.1f}s -> {next(text_encoder.parameters()).device}")

        # Transformer -> GPU0 (permanentemente, ~17GB)
        print(f"  [GPU0] Transformer (FLUX.2-dev NF4)...")
        t1 = __import__("time").time()
        transformer = Flux2Transformer2DModel.from_pretrained(
            self.cache, subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map={"": self.device0},
        )
        print(f"  {__import__('time').time()-t1:.1f}s -> {next(transformer.parameters()).device}")

        # VAE -> GPU0
        print(f"  [GPU0] VAE...")
        t2 = __import__("time").time()
        vae = AutoencoderKLFlux2.from_pretrained(
            self.cache, subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map={"": self.device0},
        )
        print(f"  {__import__('time').time()-t2:.1f}s -> {next(vae.parameters()).device}")

        print("  Tokenizer...")
        processor = AutoProcessor.from_pretrained(self.cache, subfolder="tokenizer")
        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        pipe = Flux2Pipeline(
            transformer=transformer, text_encoder=text_encoder,
            vae=vae, tokenizer=processor, scheduler=scheduler,
        )

        # Forzar _execution_device a cuda:0 (donde está transformer+VAE)
        Flux2Pipeline._execution_device = property(lambda self: device0)

        # Parchear encode_prompt: ejecutar text encoder en su GPU (cuda:1),
        # luego mover el resultado a cuda:0 (donde está el transformer)
        _original_encode_prompt = Flux2Pipeline.encode_prompt

        def _patched_encode_prompt(self, prompt, device=None, **kwargs):
            te_device = next(self.text_encoder.parameters()).device  # cuda:1
            result = _original_encode_prompt(self, prompt, device=te_device, **kwargs)
            if isinstance(result, tuple):
                result = tuple(r.to(device0) if torch.is_tensor(r) else r for r in result)
            elif torch.is_tensor(result):
                result = result.to(device0)
            return result

        pipe.encode_prompt = _patched_encode_prompt.__get__(pipe, type(pipe))

        self.pipe = pipe
        self._device0 = device0
        print("Pipeline multi-GPU listo (sin CPU offload).")
        print(f"  GPU0: transformer={torch.cuda.memory_allocated(0)/1e9:.1f}GB")
        print(f"  GPU1: text_encoder={torch.cuda.memory_allocated(1)/1e9:.1f}GB")
        return self

    def generate(self, prompt: str, seed: int, out_path: str) -> str:
        """Generate one image and save it to out_path."""
        import torch

        if self.pipe is None:
            self.load()

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        img = self.pipe(
            prompt=prompt,
            height=self.size,
            width=self.size,
            guidance_scale=self.guidance,
            num_inference_steps=self.steps,
            generator=torch.Generator(device=self._device0).manual_seed(seed),
        ).images[0]
        img.save(out_path)
        return out_path