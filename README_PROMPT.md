# Prompt cache 推理验证

在服务器上激活现有 DiffSynth 运行环境，并在本工程根目录按顺序执行下面两段脚本。路径沿用当前工程配置，运行前确认模型、LoRA 和参考图路径与服务器实际位置一致。

## 1. 导出并保存 prompt cache

这一步加载 Qwen3 文本编码器和 tokenizer，使用工程默认的固定风格迁移指令，导出 embedding 和对应的位置坐标。无需加载 DiT、VAE 或 LoRA。

```bash
MODEL_DIR="/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B"
PROMPT_CACHE="./models/style_transfer/fixed_prompt_bf16.safetensors"

CUDA_VISIBLE_DEVICES=0 python export_prompt_embedding.py \
  --model-path "$MODEL_DIR" \
  --output "$PROMPT_CACHE" \
  --device cuda \
  --dtype bfloat16
```

默认提示词为：

> Transfer the visual style of image 2 to image 1 while preserving the content, structure, objects and layout of image 1.

成功后终端会打印 `Saved ...: [1, 512, 7680], bfloat16`，并生成指定的 `.safetensors` 文件。缓存可以重复使用；导出脚本不会覆盖已有文件，重新导出时请更换 `PROMPT_CACHE` 文件名，并同步修改下一段脚本中的路径。

## 2. 使用 prompt cache 启动 infer.py

这一步只加载 DiT、VAE 和指定 LoRA，直接读取缓存，不加载文本编码器或 tokenizer。配置固定为 **4 步、CFG 1、alpha=0.5、seed=0、768×768、BF16**，使用当前默认的 step2000 checkpoint。

```bash
MODEL_DIR="/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B"
PROMPT_CACHE="./models/style_transfer/fixed_prompt_bf16.safetensors"
LORA_PATH="./models/style_transfer/FLUX.2-klein-4B_lora_StyleExpert-40k/step-2000.safetensors"
REFERENCES_DIR="/data3/jiangyikun/AIGC/StyleExpert/assets/examples"
OUTPUT_DIR="./outputs/prompt_cache_step2000_alpha05"

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model-path "$MODEL_DIR" \
  --lora-path "$LORA_PATH" \
  --prompt-embedding-path "$PROMPT_CACHE" \
  --references-dir "$REFERENCES_DIR" \
  --content-count 4 \
  --style-count 4 \
  --alpha 0.5 \
  --steps 4 \
  --cfg 1 \
  --seed 0 \
  --height 768 \
  --width 768 \
  --device cuda \
  --dtype bfloat16 \
  --warmup-runs 1 \
  --output-dir "$OUTPUT_DIR"
```

参考图目录需要包含 `content_00.png` 到 `content_03.png`，以及 `style_00.png` 到 `style_03.png`，共生成 16 个内容图 × 风格图组合。若先验证一组能否跑通，把两个 `--*-count` 都改为 `1` 即可。

不需要再传 `--prompt`，程序会使用缓存中记录的原提示词。推理的 `--dtype` 必须与导出时一致；当前缓存入口要求 `--cfg 1`。

运行后检查输出目录：

- `out_0_0.png` 等：原始生成结果。
- `out_0_0_triplet.jpg` 等：内容图、风格图、结果的并排预览。
- `infer_state.json`：每组生成是否成功及耗时。
- `timing.json`：`conditioning_mode` 应为 `cached`；`mean_pipeline_seconds` 和 `median_pipeline_seconds` 为本次成功生成的平均/中位耗时，排除了模型加载、预热和图片读写。模型加载耗时单独记录在 `pipeline_load_seconds`。

`--warmup-runs 1` 会先额外运行一次，不保存这次结果、不计入正式生成耗时。重复运行会跳过已有且校验通过的结果；重新测速或改变参数时，请使用新的 `OUTPUT_DIR`。
