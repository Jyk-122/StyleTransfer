from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from PIL import Image, ImageDraw
import time
import os

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(
        model_id="/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B",
        origin_file_pattern="tokenizer/"
    ),
)

pipe.load_lora(
    pipe.dit,
    "./models/style_transfer/FLUX.2-klein-4B_lora_StyleExpert-40k/step-2000.safetensors",
    alpha=0.5
)

prompt = "Transfer the visual style of image 2 to image 1 while preserving the content, structure, objects and layout of image 1."

output_dir = "/data3/jiangyikun/AIGC/StyleTransfer/datasets/outputs/lora_styleexpert_2000"
os.makedirs(output_dir, exist_ok=True)

total = 0

for i in range(4):
    for j in range(4):
        t = time.time()

        content_image = Image.open(
            f"/data3/jiangyikun/AIGC/StyleExpert/assets/examples/content_0{i}.png"
        ).convert("RGB")

        style_image = Image.open(
            f"/data3/jiangyikun/AIGC/StyleExpert/assets/examples/style_0{j}.png"
        ).convert("RGB")

        result_image = pipe(
            prompt=prompt,
            edit_image=[content_image, style_image],
            seed=0,
            num_inference_steps=4,
            cfg_scale=1,
            height=768,
            width=768
        )

        total += time.time() - t

        # 统一 resize，确保三张图尺寸一致
        target_size = (768, 768)
        content_vis = content_image.resize(target_size)
        style_vis = style_image.resize(target_size)
        result_vis = result_image.resize(target_size)

        # 创建并排拼接图：content | style | result
        gap = 10
        title_h = 40
        canvas_w = target_size[0] * 3 + gap * 2
        canvas_h = target_size[1] + title_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")

        # 粘贴三张图
        canvas.paste(content_vis, (0, title_h))
        canvas.paste(style_vis, (target_size[0] + gap, title_h))
        canvas.paste(result_vis, (2 * (target_size[0] + gap), title_h))

        # 可选：加标题
        draw = ImageDraw.Draw(canvas)
        draw.text((target_size[0] // 2 - 30, 10), "Content", fill="black")
        draw.text((target_size[0] + gap + target_size[0] // 2 - 20, 10), "Style", fill="black")
        draw.text((2 * (target_size[0] + gap) + target_size[0] // 2 - 25, 10), "Result", fill="black")

        canvas.save(f"{output_dir}/out_{i}_{j}_triplet.jpg", quality=95)

print(total / 16)
