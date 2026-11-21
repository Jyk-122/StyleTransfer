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

## 3. 独立控制生成图、内容图和风格图尺寸

训练入口和 `infer.py` 都支持下面的参数，文本输入模式与 prompt cache 模式通用。新参数同时接受连字符和下划线写法，例如 `--style-height` 与 `--style_height` 等价。

| 图像角色 | 固定尺寸参数 | 按原图比例限制像素数 |
| --- | --- | --- |
| 生成图；训练时对应监督目标 `image` | `--height`、`--width` | 训练可省略宽高，使用原有 `--max_pixels`；推理仍指定宽高 |
| 内容参考 `edit_image[0]` | `--content-height`、`--content-width` | `--content-max-pixels` |
| 风格参考 `edit_image[1]` | `--style-height`、`--style-width` | `--style-max-pixels` |

- 固定宽高必须成对指定，且都是正的 16 的倍数。处理方式为等比例缩放后中心裁剪；若目标宽高比与原图不同，会裁掉部分边缘。
- 参考图的 `max-pixels` 与其固定宽高互斥：保留原图比例缩小，宽高向下对齐到 16，不放大小图。对齐可能产生少量边缘裁剪。
- 启用任一参考尺寸参数后，参考图直接从原图独立处理，训练目标的尺寸不会覆盖参考尺寸；pipeline 自动缩放会关闭，避免重新放大到 1024²。未配置的另一类参考图仍使用约 1024² 面积、保持原图比例的默认规则。
- 不传任何新的参考尺寸参数时，沿用原有处理路径，包括训练中通用数据预处理的行为。

例如：生成图/训练目标为 `768×768`，内容图为 `1024×1024`，风格图为 `512×512`。先确认 `train.sh` 中的数据和模型路径，再从工程根目录运行：

```bash
bash train.sh \
  --height 768 --width 768 \
  --content-height 1024 --content-width 1024 \
  --style-height 512 --style-width 512 \
  --output_path ./models/style_transfer/FLUX.2-klein-4B_lora_c1024_s512_o768
```

`train.sh` 会把末尾参数传给训练入口，因此也可以覆盖脚本中的同名参数。原来的数据路径、学习率、prompt cache 等设置仍由脚本提供。

对应推理命令如下，路径变量沿用上文：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model-path "$MODEL_DIR" \
  --lora-path "$LORA_PATH" \
  --prompt-embedding-path "$PROMPT_CACHE" \
  --references-dir "$REFERENCES_DIR" \
  --height 768 --width 768 \
  --content-height 1024 --content-width 1024 \
  --style-height 512 --style-width 512 \
  --steps 4 --cfg 1 --alpha 0.5 --seed 0 \
  --dtype bfloat16 --warmup-runs 1 \
  --output-dir ./outputs/prompt_cache_c1024_s512_o768
```

推理训练后的新 checkpoint 时，将 `LORA_PATH` 改为新训练输出目录中的实际权重文件。若希望参考图保持原图比例，可把两组参考宽高替换为 `--content-max-pixels 1048576 --style-max-pixels 262144`。生成图的宽高仍单独设置。

仅改变图像尺寸无需重新导出 prompt cache。尺寸设置会记录在推理配置和 manifest 中，改变尺寸请使用新的输出目录。计时包含参考图缩放，与原先在 pipeline 内缩放的计时范围一致。

这些参考尺寸参数用于原始图片训练或 `:data_process` 预处理；已有 `:train` 预处理数据中的 latent 尺寸已固定，要换尺寸需重新预处理。

## 4. 分项计时与风格 VAE cache 时延估计

在现有推理命令中增加 `--profile-components` 即可。它只统计耗时，风格图仍会在每次生成时正常编码，没有实现或模拟 VAE cache，也不改变图像预处理和模型计算。

例如，沿用第 2 节的路径变量，以固定一张风格图、四张内容图进行 4 步、CFG 1 测量：

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model-path "$MODEL_DIR" \
  --lora-path "$LORA_PATH" \
  --prompt-embedding-path "$PROMPT_CACHE" \
  --references-dir "$REFERENCES_DIR" \
  --content-count 4 --style-count 1 \
  --height 768 --width 768 \
  --steps 4 --cfg 1 --alpha 0.5 --seed 0 \
  --dtype bfloat16 --device cuda \
  --warmup-runs 1 --profile-components \
  --output-dir ./outputs/profile_components
```

不传 `--prompt-embedding-path` 也能计时，此时文本编码会计入 `other`。不传 `--profile-components` 时，保留原来的总时间统计。

终端会显示各项平均耗时和占比。详细结果在 `timing.json` 的 `component_profile` 中：

| `mean_seconds` 字段 | 测量范围 |
| --- | --- |
| `content_vae` | 内容图的 `vae.encode` |
| `style_vae` | 风格图的 `vae.encode` |
| `dit` | 所有 DiT forward 的累计时间；4 步、CFG 1 时为 4 次 |
| `decode` | 最终生成 latent 的 `vae.decode` |
| `other` | 总时间减去上述四项，包括预处理、文本条件、调度、调用外的数据搬运等 |
| `total` | 一次 pipeline 调用；不含模型加载、预热、参考图文件读取和结果保存 |

`percent_of_total` 给出各项平均耗时占总时间的百分比。`samples` 和 `infer_state.json` 中的 `components` 保存逐组计时、DiT 每次 forward 耗时、VAE 实际输入的 `[B,C,H,W]`、风格 latent 的形状、dtype 和字节数。只汇总本次新生成成功的组合，失败或跳过的组合不计入。

CUDA 在计时边界同步，防止把异步计算的提交时间当成执行时间。同步和计时包装本身会带来额外开销，因此该模式用于组件占比分析；普通推理延迟可另开输出目录、不加此开关测量。预热针对首个待生成组合；其他输入形状的首次调用仍可能包含初始化开销。

`style_vae_cache_estimate` 是缓存已命中时的理想估计：

```text
预计单次耗时 = 实测总耗时 - 实测风格 VAE 编码耗时
预计节省比例 = 风格 VAE 编码耗时 / 总耗时
预计加速倍数 = 总耗时 / (总耗时 - 风格 VAE 编码耗时)
```

估计假设缓存读取/传输开销为零，只扣除风格 VAE 编码，不扣除风格 token 在 DiT 中的计算，也不额外扣除预处理。第一次构建缓存仍需编码；同一风格复用 N 次时，忽略读取成本的平均估计为 `T_total - (1 - 1/N) * T_style_vae`。实际部署收益需实现 cache 后再测。

当前 FLUX.2 VAE 的 `encode` 使用预测均值，没有随机采样。在风格图及预处理、VAE 权重、编码精度都相同，且缓存保留原 latent 数值的条件下，复用它不会引入额外质量损失；这里不意味着 VAE 压缩本身无损。
