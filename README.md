<h1 align="center">
  <img src="assets/logo.jpg" alt="InternVL-U Logo" width="80"><br>
  InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing
</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-2603.09877-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09877)&nbsp;
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-InternVL--U-yellow)](https://huggingface.co/InternVL-U/InternVL-U)&nbsp;
[![GenEditEvalKit](https://img.shields.io/badge/GitHub-GenEditEvalKit-181717?logo=github)](https://github.com/open-compass/GenEditEvalKit)&nbsp;
[![TextEdit Benchmark](https://img.shields.io/badge/GitHub-TextEdit%20Benchmark-181717?logo=github)](https://github.com/open-compass/TextEdit)

Shanghai AI Laboratory, InternVL-U Team

Welcome to the official repository for InternVL-U project!
If you find our work helpful, please give us a ⭐.
</div>

## 🎉 News
- **[2026/03/06]** 🔥InternVL-U **technical report** released.  Check it out on [[arXiv]](https://arxiv.org/abs/2603.09877). 
- **[2026/03/06]** ✨**Inference code and model checkpoint** released. Check it out on [[Hugging Face]](https://huggingface.co/InternVL-U/InternVL-U).
- **[2026/03/06]** 🛠️ **GenEditEvalKit** released — a unified evaluation toolkit for multimodal image generation and editing models, designed to help developers efficiently manage inference and evaluation across numerous benchmarks for the unified multimodal model (UMM) and image generation and editing models.  Check it out on [[GitHub]](https://github.com/open-compass/GenEditEvalKit).
- **[2026/03/06]** 📝 **TextEdit Benchmark** released — a high-quality, multi-scenario benchmark for evaluating text editing capabilities in image generation models. Try it out and see how well your model performs on challenging text editing tasks~. Check it out on [[GitHub]](https://github.com/open-compass/TextEdit).
- **[2026/03/19]** We now support multi-image understanding inference. Use the examples from Quick Start.
## 📖 Introduction

**InternVL-U** is a **4B-parameter unified multimodal model (UMM)** that brings **multimodal understanding, reasoning, image generation, image editing** into a *single* framework, aiming to **democratize omni-capable multimodal intelligence** with an efficient and practical model size.

### Key Highlights
- **Unified yet modular design**: built on **unified contextual modeling** with **modality-specific modularity** and **decoupled visual representations**.
- **Strong backbone + strong generator**: integrates a **state-of-the-art MLLM** with a specialized **MMDiT-based visual generation head**.
- **High-quality data synthesis**: a comprehensive data pipeline for **high-semantic-density tasks** (e.g., text rendering and editing, scientific reasoning) using **Chain-of-Thought (CoT)** to align *abstract intent* with *precise visual execution*.
- **Performance–efficiency win**: within a limited parameter scale, InternVL-U outperforms unified open-source UMM baselines in generation and editing, while **retaining strong multimodal understanding and reasoning**.

We hope **InternVL-U** serves as a **strong baseline** and accelerates progress toward **comprehensive, AGI-oriented omni-capable UMMs**.
<p align="center">
  <img src="assets/teaser1.jpg" width="45%">
  <img src="assets/teaser2.jpg" width="45%">
</p>

## ⚡Quick Start


Before getting started, make sure you have installed all required dependencies.
```
pip install -r requirements.txt
```
Model checkpoint is available on [Hugging Face](https://huggingface.co/InternVL-U/InternVL-U).

### Inference Demo
We provide the following demos to showcase InternVL-U’s unified pipeline for multimodal understanding, image generation, and image editing, with optional reasoning-guided (text+image) outputs.

**Tips**: During image generation or editing, if `generation_mode` is set to `"image"`, the CoT text reasoning mode will not be activated. We recommend this mode for most scenarios, especially simple scenes and instructions, as it can already produce good results. When the user input involves more challenging reasoning, we recommend setting `generation_mode` to `"text_image"` to enable more complex generation.

<details>
<summary><b>Generate Text</b> - Click to expand</summary>

```python
import torch
from PIL import Image
from internvlu import InternVLUPipeline

prompt = "What is the amino acid shown in the picture?"
image = Image.open("assets/amino_acid.png").convert("RGB")

pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

tokenizer = pipeline.processor.tokenizer

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        image=image,
        max_new_tokens=1024,
        generation_mode="text",
    ).generate_output[0]

print(tokenizer.decode(output, skip_special_tokens=True))
```

</details>

<details>
<summary><b>Generate Text with Multi Images</b> - Click to expand</summary>

```python
import torch
from PIL import Image
from internvlu import InternVLUPipeline

prompt = "Tell me the difference of the two pictures."
images = [
[Image.open("assets/logo.jpg").convert("RGB"),
Image.open("assets/logo_zh.jpg").convert("RGB"),]
]


pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

tokenizer = pipeline.processor.tokenizer

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        image=image,
        max_new_tokens=1024,
        generation_mode="text",
    ).generate_output[0]

print(tokenizer.decode(output, skip_special_tokens=True))
```

</details>

<details>
<summary><b>Generate Image</b> - Click to expand</summary>

```python
import torch
from internvlu import InternVLUPipeline

prompt = """In the deep indigo night sky, a grand fireworks festival is at its peak, with countless dazzling Mars arranged precisely, condensed into the huge and dazzling "InternVL-U" words. The letters are composed of highly saturated electric blue and dreamy purple fluorescent particles, presenting a futuristic streamlined font surrounded by scattered golden fragments resembling stardust, and the final "U" gives off a fluid metallic texture. Below is a brightly lit modern city, with the shimmering sea perfectly reflecting this stunning scene. Amidst the swirling smoke, it showcases the ultimate visual allure of technology and romance intertwined."""

pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

with torch.no_grad():
    image = pipeline(
        prompt=prompt,
        generation_mode="image",
        height=576,
        width=1024,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

image.save(f"example_t2i.png")
```
</details>

<details>
<summary><b>Generate Image with Reasoning-guided</b> - Click to expand</summary>

```python
import torch
from internvlu import InternVLUPipeline

prompt = """生成一张展现希望正在生长、充满力量感的画面。"""

pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

tokenizer = pipeline.processor.tokenizer

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        generation_mode="text_image",
        generator=torch.Generator(device="cuda").manual_seed(42)
    )
    output_text = output.generate_output[0]
    image = output.images[0]

print(tokenizer.decode(output_text, skip_special_tokens=True))
image.save(f"example_guided_t2i.png")
```

</details>

<details>
<summary><b>Edit Image</b> - Click to expand</summary>

```python
import torch
from PIL import Image
from internvlu import InternVLUPipeline

prompt = """将书生置身于一个充满春节氛围的温暖室内空间中，房间内张灯结彩，窗外远处有灯笼，窗边装饰着精致的剪纸，红金配色的彩带在室内轻轻垂落。背景为柔和温暖的橙黄色灯光和柔软的沙发，营造出温馨而治愈的春节夜晚氛围。书生双手捧着一个红包，红包上有金色的边框，脸上洋溢着抑制不住的开心笑容，浓眉大眼闪闪发亮，流露出惊喜与幸福的神情。他身体微微前倾，肩膀自然放松，仿佛忍不住想要展示这份喜悦。整个场景充满温馨、喜庆与治愈感，在柔和的暮色与节日装饰映衬下，展现出春节特有的欢乐与人情味。"""
input_image = Image.open("assets/intern.png").convert("RGB")

pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

with torch.no_grad():
    image = pipeline(
        prompt=prompt,
        image=input_image,
        generation_mode="image",
        height=input_image.size[1],
        width=input_image.size[0],
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

image.save(f"example_edit.png")
```

</details>

<details>
<summary><b>Edit Image with Reasoning-guided</b> - Click to expand</summary>

```python
import torch
from PIL import Image
from internvlu import InternVLUPipeline

prompt = """Convert into the style of a Valentine's Festival picture, suitable for use as a profile picture on social media."""
input_image = Image.open("assets/lines_puppy.png").convert("RGB")

pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

tokenizer = pipeline.processor.tokenizer

with torch.no_grad():
    output = pipeline(
        prompt=prompt,
        image=input_image,
        num_beams=5,
        generation_mode="text_image",
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    output_text = output.generate_output[0]
    image = output.images[0]

print(tokenizer.decode(output_text, skip_special_tokens=True))
image.save(f"example_guided_edit.png")
```

</details>



<details>
<summary><b>Batch Inference</b> - Click to expand</summary>

```python
import torch
from PIL import Image
from internvlu import InternVLUPipeline

prompts = [
    "Segment the little boy",
    "Provide the bounding box for the little boy"
]
input_images = [Image.open("assets/panda_and_boy.png").convert("RGB")] * 2


pipeline = InternVLUPipeline.from_pretrained(
    "/path/to/internvl-u-checkpoint",
    torch_dtype=torch.bfloat16,
)

pipeline.to("cuda")

with torch.no_grad():
    output_images = pipeline(
        prompt=prompts,
        image=input_images,
        generation_mode="image",
        height=input_images[0].size[1],
        width=input_images[0].size[0],
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images

for idx, image in enumerate(output_images):
    image.save(f"example_edit_{idx}.png")
```

</details>




## Citation
If you find our InternVL-U useful, please cite our InternVL-U technical report using this BibTeX.

```bibtex
@article{tian2026internvl,
  title={InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing},
  author={Tian, Changyao and Yang, Danni and Chen, Guanzhou and Cui, Erfei and Wang, Zhaokai and Duan, Yuchen and Yin, Penghao and Chen, Sitao and Yang, Ganlin and Liu, Mingxin and others},
  journal={arXiv preprint arXiv:2603.09877},
  year={2026}
}
```

## 🙏 Acknowledgement

We sincerely thank the contributors of the following open-source projects for their valuable code, models, and datasets. **InternVL-U** is built upon and inspired by these outstanding works:

InternVL3.5, BAGEL, Qwen2.5-VL, Qwen3-VL, Qwen-Image, BLIP3-o, OpenGPT-4o-Image, ShareGPT-4o-Image, OmniGen2, UniWorld-V1, PicoBanana, Nano-Consist, and NHR-Edit, and so on. 
