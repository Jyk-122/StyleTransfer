# 风格迁移评测

面向本工程的 `内容图 + 风格参考图 → 模型生成图`。评测与生成分开，支持直接评已有图片；默认生成配置是 **FLUX.2-klein-4B、4 步、CFG 1**。评测模型用于离线选 checkpoint/alpha，不放进端侧推理链路。

## 指标定义

| 输出列 | 比较对象 | 定义 / 范围 | 越大越好 |
|---|---|---|---|
| `dino_content` | 生成图、内容图 | DINOv2 最后一层 CLS 特征 L2 归一化后的余弦相似度，理论范围 [-1, 1] | 是 |
| `csd_style` | 生成图、风格图 | CSD **风格投影头**特征的余弦相似度，理论范围 [-1, 1] | 是 |
| `qwen_content` | 三张图联合判断，重点比较生成图与内容图 | 固定评分标准，0–10 | 是 |
| `qwen_style` | 三张图联合判断，重点比较生成图与风格图 | 固定评分标准，0–10 | 是 |

Qwen 同时给出理由、两张图中的人物数量估计、判断不确定性，以及漏内容、多内容、布局变化、畸变、风格图内容混入、仅改色调、非预期模糊等标记。它看不到 checkpoint 名称、alpha、数据集目标图或其他实验分数。评分标准在 [`qwen_prompt.py`](qwen_prompt.py)，修改后缓存会失效。

不把这四个指标直接加成一个“总分”。DINO 0.85 **不是** 85% 内容保留率；Qwen 8 分也不是 80% 准确率。Qwen 是自动评审，有偏差，人物数量也只是估计。先人工复核一批多人、黏土、简笔画等样本，确认分数和人工判断一致，再用于大规模筛选。这里的双轴 0–10 评分是本项目自定义协议，不是复现 StyleExpert 原文的二元 Qwen semantic score。

## 安装

在仓库根目录执行以下命令。建议给评测单独建 Python 3.10+ 环境；生成命令仍可用现有 DiffSynth 训练环境。先安装适合服务器 CUDA 的 torch/torchvision，再安装其余依赖，避免无意改变训练环境。

```bash
pip install -r eval/requirements.txt
```

三个入口都采用模块方式运行：`python -m eval.generate`、`python -m eval.run`、`python -m eval.compare`。执行 `--help` 不加载模型。

默认评测模型：

- DINOv2：`facebook/dinov2-base`，支持 `--dino-model /本地目录`。
- CSD：自动下载 `tomg-group-umd/CSD-ViT-L/pytorch_model.bin`；也可用 `--csd-weights /路径/pytorch_model.bin`。使用官方学习到的视觉骨干及 `last_layer_style`，严格检查参数；不使用普通 CLIP 代替，也不需要额外下载 CLIP 初始权重。支持带 `model_state_dict` 的官方 checkpoint、纯 state dict 和对应的 safetensors。
- Qwen：默认本地 `Qwen/Qwen3-VL-8B-Instruct`，支持 `--qwen-model /本地目录`。`--qwen-device-map auto` 可交给 Accelerate 分配显存；也支持独立 API 服务，见下文。

离线机器提前下载模型，传本地路径并加 `--local-files-only`。联网模式可通过 `--dino-revision`、`--csd-revision`、`--qwen-revision` 固定 Hugging Face commit。CSD checkpoint 按 SHA256 记录；HF 模型记录解析到的 revision；本地模型记录配置哈希和权重文件大小/修改时间清单的指纹，**不是完整权重校验和**。不要在评测过程中覆盖模型文件。

## 接入当前 `infer.py`

根目录 `infer.py` 按内容图 × 风格图生成所有组合，默认保留原来的 4×4、step2000、alpha=0.5、4 步、CFG 1。现在每组输出：

```text
输出目录/
├── out_0_0.png              # 原始生成图：无缩放、无标题、无拼接，PNG 无损保存
├── out_0_0_triplet.jpg      # 内容 | 风格 | 结果预览，沿用原文件名
├── out_0_1.png
├── out_0_1_triplet.jpg
├── ...
├── manifest.jsonl          # 显式绑定内容图、风格图、单独的结果 PNG
├── infer_config.json       # checkpoint、alpha、steps、seed、参考图哈希等
└── infer_state.json        # 生成状态、结果哈希、耗时；支持续跑
```

旧版本只有 `*_triplet.jpg`，不能把整张拼图当作生成结果送入评测。重跑新版会补齐单独的 PNG 和清单。`--output-pattern` 的单一 metadata 索引不适合这个双索引网格，直接用自动生成的 `--manifest`。

先把参考图放到 `--references-dir`，默认命名 `content_00.png`…`content_09.png` 和 `style_00.png`…`style_09.png`，再运行：

```bash
python infer.py \
  --content-count 10 --style-count 10 \
  --alpha 0.5 --steps 4 --cfg 1 --seed 0 \
  --output-dir /data3/jiangyikun/AIGC/StyleTransfer/datasets/outputs/lora_styleexpert_2000_a05

python -m eval.run \
  --manifest /data3/jiangyikun/AIGC/StyleTransfer/datasets/outputs/lora_styleexpert_2000_a05/manifest.jsonl \
  --output-dir /data3/jiangyikun/AIGC/StyleTransfer/datasets/outputs/lora_styleexpert_2000_a05/scores \
  --run-name step2000_alpha0.5 \
  --cache-dir eval/artifacts/shared_cache \
  --metrics dino csd qwen \
  --qwen-model Qwen/Qwen3-VL-32B-Instruct \
  --qwen-dtype bfloat16 --qwen-device-map cuda:0
```

支持 `--lora-path`、`--references-dir`、`--content-pattern`、`--style-pattern`、`--no-lora`；命名模板的索引字段是 `{index}`，例如 `'content_{index:02d}.png'`，超过 9 时得到 `content_10.png`。可用 `--plan-only` 先检查输入并写清单。新 checkpoint/alpha/seed/样本规模使用不同输出目录，避免实验互相覆盖。预览按比例缩放留白，评测始终读取原始结果。

## 90GB 显存的评委模型建议

当前工程可以直接采用 **Qwen3-VL-32B-Instruct，BF16**。其官方页面列出约 33B 参数；按每参数 2 字节估算，纯权重约 66GB（约 62GiB），另需 KV cache、视觉特征和运行缓冲。按单卡约 90GB 可用显存、每次一组三张图、最长边 1024、输出上限 1024 tokens 估计可行，实际峰值需要在服务器验证。生成进程结束后再评测，避免与训练/FLUX 同时占用显存。[官方模型与用法](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)

代码的通用默认仍是 8B，以上命令显式选 32B。所有待比较的 checkpoint 使用同一评委、同一 revision 和同一评分标准。先人工标注 30–50 个含正确结果、漏人物、内容混入、仅调色等情况的样本，检查评委判断是否符合你的标准；更大的参数量本身不构成风格评审准确率的验证。

如果要比较更新的 **Qwen3.8-27B**，可在独立的、支持该模型的 vLLM 环境部署后走 API；当前固定的 Transformers 4.57.6 / `Qwen3VLForConditionalGeneration` 本地后端只用于 Qwen3-VL，不能直接换成 3.8 的模型名。对于兼容 vLLM 的接口，用 `--qwen-disable-thinking` 发送 `chat_template_kwargs={"enable_thinking":false}`，以获得直接的 JSON 答案。这个候选需要同样的人工校准，未在本工程做真实权重对比。[Qwen3.8 官方部署与非思考模式](https://huggingface.co/Qwen/Qwen3.8-27B)

## 评测集规模建议

以下是本项目的工程起点，不是统计充分性的保证：

| 用途 | 内容图 × 风格参考图 | 每个 checkpoint/alpha 的生成数 |
|---|---|---|
| 快速回归、定位当前退化 | 10 × 10 | 100，固定 1 个 seed |
| 选择 checkpoint/alpha 的验证集 | 30–50 × 10 | 300–500，固定 1 个 seed |
| 参数选定后的独立测试 | 50 × 20 | 全组合 1000；也可均衡抽取约 500 对 |

10×10 共享 10 个内容场景和 10 张风格图，**不等于 100 个独立内容样本**。从中挑出的“最好 checkpoint”仍是验证集选择结果；最终测试应换未参与选择的内容图和风格参考图。StyleExpert 示例图适合作为排查素材，未确认其来源/复用关系前，不把它们称为严格的未见测试集。

初始 10 张内容图可安排 4 张多人场景（2/3/4/5 人、不同遮挡/构图）、2 张单人全身/半身、2 张动物或多物体、2 张复杂场景。10 张风格参考覆盖简笔画、黏土、素描、水彩、油画、像素等不同视觉特征，避免只是同种画风换色。正式测试中，同一风格类别选至少两张不同内容的参考图，以观察是否复制参考图内容。

先在固定 seed 上筛选，最终 2–3 个候选再用 3 个固定 seeds 检查稳定性。报告总体均值之外也看多人/遮挡等内容子组、每种风格和遗漏率。重复 seeds 与全组合样本之间有相关性；若以后报告置信区间，应按内容/风格分组处理，而不是把所有生成图视作独立样本。

## 1. 生成固定评测集的结果

直接读取现有 DiffSynth `metadata.json`：`edit_image[0]` 是内容，`edit_image[1]` 是风格，`image` 是数据集目标。**`image` 不会被当成模型生成图。**

建议先准备独立验证集的 metadata。下面从该文件固定随机抽取 100 条，不是自动做训练/验证切分；不足 100 条则全部使用。每次保持 metadata、`--limit`、`--selection-seed`、`--seed` 不变。

```bash
python -m eval.generate \
  --metadata-path /data/StyleExpert-40K-val/metadata.json \
  --dataset-base-path /data/StyleExpert-40K-diffsynth \
  --model-path /data3/jiangyikun/AIGC/models/FLUX.2-klein-4B \
  --lora-path models/style_transfer/FLUX.2-klein-4B_lora/step-18000.safetensors \
  --alpha 0.5 --steps 4 --cfg 1 --height 768 --width 768 \
  --limit 100 --selection-seed 42 --seed 0 \
  --output-dir eval/artifacts/step18000_a05
```

输出包括 `images/`、`manifest.jsonl`、`generation.json` 和 `generation_state.json`。图片名使用**原 metadata 索引**，如 `000123.png`，不会因抽样重新编号。`--limit 0` 使用全部记录，`--plan-only` 只检查参考图并生成清单，不加载生成模型。

换 step2000、alpha=1/0.5 等配置时，使用新的 `--output-dir`。**每个命令只加载一套 checkpoint/alpha，启动一个全新的 pipeline**，避免本工程 `load_lora` 融合权重时跨实验叠加 LoRA。省略 `--lora-path` 可生成无 LoRA 对照组。

相同命令可续跑：已生成且哈希吻合的图片跳过；参数、参考图或实现变动时拒绝混写。`generation_state.json` 保留每条生成错误及 pipeline 耗时；耗时含该次 pipeline 调用，不含加载模型和读写文件，包含首次调用开销，不能据此声称端侧实时性能已达标。

默认输出 768×768；对非方形内容图，选择适合内容比例的 `--height/--width`（16 的倍数），避免把构图比例变化误判为模型退化。比较实验应保持这些配置一致。

## 2. 打分

```bash
python -m eval.run \
  --manifest eval/artifacts/step18000_a05/manifest.jsonl \
  --output-dir eval/artifacts/step18000_a05/scores \
  --run-name step18000_alpha0.5 \
  --cache-dir eval/artifacts/shared_cache \
  --metrics dino csd qwen \
  --device cuda --batch-size 8 \
  --qwen-device-map auto
```

可先加 `--check-only` 检查全部内容、风格、输出图片的完整解码和路径，不加载打分模型。默认逐个加载和释放 DINO、CSD、Qwen，减少同时占用显存；Qwen 每次评一组三张图片。

按需使用 `--metrics dino csd` 或 `--metrics qwen`。**每次报告只包含本次指定的指标**；重跑全部指标时可复用缓存。相同报告目录只能用于同一组样本和生成配置。

已有生成结果也可以直接接入，不必重新生成：

```bash
python -m eval.run \
  --metadata-path /data/val/metadata.json \
  --dataset-base-path /data/StyleExpert-40K-diffsynth \
  --outputs-dir /data/results/step18000_a05 \
  --output-pattern '{index:06d}.png' \
  --output-dir eval/artifacts/existing_scores \
  --metrics dino csd qwen
```

这里要求文件名与原 metadata 索引一一对应。若你的命名不同、样本顺序不同，写一个显式 JSONL 清单最可靠；每行如下，相对路径均以**清单所在目录**为根：

```json
{"id":"people_003_clay","content":"inputs/content.png","style":"inputs/clay.png","output":"results/prediction.png","seed":0,"style_group":"clay"}
```

`id` 在一个清单内唯一；不同实验使用相同 id 对应相同内容/风格/seed。同一图对的不同 seed 使用不同 id。可选 `target` 仅保留来源，不参与评分；`style_group` 是人工或已有可靠标签，不会从文件名猜测。已有图片没有声明的 seed/生成参数无法从图片反推，比较前由你确认一致；需要记录时在清单添加 `seed` 和 `generation` 对象。

### Qwen 使用 API

可接自建 vLLM/OpenAI-compatible Qwen 服务，不必在打分进程加载 8B 模型。服务需接受三张图片及 `seed`、`temperature`、`max_tokens`；vLLM 部署需允许每条请求至少 3 张图，例如设置 `--limit-mm-per-prompt '{"image":3}'`（参见 [vLLM 多图配置](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)）。实际服务版本与量化设置写入 `--qwen-api-revision`，避免同名模型更新后继续复用旧缓存。

```bash
python -m eval.run \
  --manifest eval/artifacts/step18000_a05/manifest.jsonl \
  --output-dir eval/artifacts/step18000_a05/scores \
  --cache-dir eval/artifacts/shared_cache \
  --metrics dino csd qwen \
  --qwen-backend api \
  --qwen-base-url http://localhost:8000/v1 \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --qwen-api-revision my-fixed-qwen3vl-deployment-v1
```

鉴权 key 从 `QWEN_API_KEY` 环境变量读取，或用 `--qwen-api-key-env` 指定其他变量名。key 不写入报告。API 模式会向你指定的端点发送三张图片；默认本地模式不调用外部推理 API。服务支持 JSON mode 时可加 `--qwen-json-mode`。

本地 greedy 解码，API temperature=0、seed=0；这减少随机性，但不保证跨硬件/服务完全一致。三张图均保持宽高比、不裁剪，默认最长边不超过 1024；Qwen processor 还会按模型要求处理尺寸。修改 `--qwen-max-side` 会改变评测协议。返回 JSON 不合法或请求失败默认重试一次，原始响应与错误均保留；失败不会记成 0 分。

## 3. 对比 checkpoint / alpha

```bash
python -m eval.compare \
  eval/artifacts/step2000_a05/scores \
  eval/artifacts/step18000_a05/scores \
  eval/artifacts/step18000_a10/scores \
  --output-dir eval/artifacts/comparison
```

生成 `comparison.csv` 和 `comparison.json`。每个指标在**所有实验均成功评分的同一批样本**上求均值，另报覆盖率和相对第一个实验的配对平均差值。输入图、id、prompt、seed 不匹配或打分模型/预处理/评分标准不一致时拒绝直接比较；只有确实需要比较公共子集时使用 `--intersection`。不同生成 steps/alpha/分辨率可以作为实验变量，表内会记录，不自动宣称它们属于相同推理配置。

建议当前优先固定 4 步、CFG 1，对照 step2000/step18000 × alpha=0.5/1。关注在风格分提高时内容分是否下降，以及 Qwen 是否报告人物遗漏、畸变；不要仅按单列最大值选 checkpoint。用验证集选参数，另留测试集；StyleExpert 中图片会跨风格/目标角色复用，随机切行不保证无泄漏。

## 报告、缓存与失败

- `scores.jsonl`：逐条分数、Qwen 完整理由/标记/原始响应、错误、输入图片 SHA256、生成配置。
- `scores.csv`：方便排序、筛选和人工核对的扁平表。
- `summary.json`：均值、中位数、样本标准差、有效数/总数/覆盖率；有 `style_group` 时另报各组均值及组间宏平均。无分数为 `null`，不填 0。
- `config.json`：模型、权重身份、预处理、完整评分标准、依赖版本、代码指纹。
- `input_audit.json`：所有输入文件的读取结果、尺寸和字节哈希。
- 缓存按**指标协议指纹 + 图片字节哈希**组织，可跨 checkpoint 共享参考图特征；改图、换权重/预处理/评分标准均不会命中旧缓存。失败记录会重试。完整续跑仍会初始化评测模型，但已缓存图片/三元组不重复前向。

任一请求的指标失败时命令返回非零，成功指标的报告仍写出。DINO/CSD 批次失败时该批次保留错误；显存不足可降低 `--batch-size` 后续跑。对比表只使用公共且有效的样本，务必同时检查 `*_n` 和覆盖率，避免只看小量幸存样本的均值。不要同时向同一报告/生成目录启动多个进程。

DINO 和 CSD 使用各自官方预处理中的中心裁剪，画面边缘可能没被特征覆盖；人物计数等细节需要结合看完整图的 Qwen 和人工核对。CSD 官方仓库目前提示公开权重分数与论文存在差异，因此这里用于**固定权重下的相对比较**，不直接与论文表格绝对数值对标。[CSD 官方实现](https://github.com/learn2phoenix/CSD)

## 验证代码

```bash
python -m unittest discover -s eval/tests -v
```

使用小型随机本地模型、合成图片和模拟 API：验证 DINO 的 CLS 特征、CSD 风格头与原计算路径一致、checkpoint 加载、本地 Qwen 三图 processor/generate 接口、图像顺序、评分解析、缓存失效、缺图覆盖率、生成续跑和实验比较。无需下载正式权重。它们验证实现接口和数据流，不能证明真实评分与人类偏好一致；正式环境仍应先跑 5–20 条样本进行人工校准。

参考：[DINOv2 模型接口](https://huggingface.co/facebook/dinov2-base)、[CSD 模型与论文](https://huggingface.co/tomg-group-umd/CSD-ViT-L)、[Qwen3-VL 官方用法](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。
