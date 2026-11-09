# Backbone 对比规划

> 目标：判断换生成骨干能否在下游检测上带来稳定收益；只有收益明确才替换 SD1.5。
> 适用阶段：可见性/形状先验的第一轮结论出来之后。

## 1. 候选与判断

| 骨干 | 本地权重 | 与现有 adapter 兼容性 | 预期收益 | 成本 | 结论 |
| --- | --- | --- | --- | --- | --- |
| SD1.5 base（现状） | `newproject/pretrained/stable-diffusion-v1-5-full` | 原生 | 基线 | 低 | 保留为对照 |
| SD2 inpainting | HF 缓存已完整（9 通道 UNet） | 需要 9 通道输入改造（已实现 `v07/model.py`） | 中高：inpainting 先验与"掩码内补全"任务天然匹配 | 中 | **第一优先** |
| SD2.1 base | 未缓存，需下载 ~5GB | UNet 结构与 SD1.5 相同，可近似 drop-in；文本维度 1024 | 中：更好的文本/图像先验 | 中 | 第二优先 |
| SDXL base/inpaint | 未缓存 | UNet 通道结构与 CLIP 双编码器不同，adapter 需重写 | 未知 | 高 | 暂缓 |
| FLUX / SD3.5 | 未缓存 | DiT/MMDiT，adapter 完全不兼容 | 未知 | 很高 | 不做 |

**判断**：先把 SD2-inpainting 跑通。它不是"更大的模型"，而是"任务结构更匹配的模型"——9 通道输入显式接收（带洞图像, mask），与我们的 mask 条件生成完全一致，边界一致性可能更好，且不用重新设计 adapter。

## 2. 已完成的代码支持

- `v07/backbones.py`：`sd15` 与 `sd2-inpaint` 注册、加载、校验（9 通道、512px）。
- `v07/model.py`：`V07InfillPredictor`，inpaint 时按 diffusers 约定拼 `(noisy, mask, masked_image_latents)`。
- `v07/render.py`：`denoise_masked_infill_v07` 与 `V07ExactMaskRenderer`，推理时把正常图 + 洞编码成 `masked_image_latents`。
- `v07/objective.py`：ranking 负样本（冻结 normal U-Net）也支持 9 通道输入。
- 生成阶段通过 `v07/generate.py` 的上下文管理器替换 renderer，不改 v06。

**尚未验证**：SD2-inpaint 权重能否在本机 CPU 完整加载（下一步用 `--device cpu` 做一次加载冒烟）。

## 3. 实验

| 编号 | 实验 | 对照 | 判定 |
| --- | --- | --- | --- |
| B1 | SD2-inpaint，无可见性损失 | SD1.5 同协议 | 4 类中至少 3 类 AP-P 不降，且 2 类提升 ≥2 |
| B2 | SD2-inpaint + 可见性损失 | B1 / 可见性单独 | 是否叠加 |
| B3 | SD2.1 base（若下载） | SD1.5 | 仅当 B1 收益不明确时再做 |
| B4 | 骨干 × 训练步数（600/1200/2000） | 各自 | 排除"只是训得更久" |

## 4. 成本估算

- SD2-inpaint UNet 3.4GB fp32 / 1.7GB fp16，VAE 与 SD1.5 相同。
- 训练 batch=1、512、adapter 残差：显存约 12–16GB，3090 可跑。
- 单类训练 1200 步与基线同量级（约 0.5–1 GPU·h）；生成 200 对约 0.7–1 GPU·h。
- 4 类一轮约 6–8 GPU·h，双卡一天可跑 2 个骨干变体。

## 5. 风险

- SD2-inpaint 的文本编码器是 OpenCLIP ViT-H，prompt 语义与 SD1.5 不同；`normal_prompt` 可能需要在 SD2 上重新调（先沿用）。
- 9 通道输入会让训练显存增加约 1–2GB；如果 OOM，先降 `latent_size` 不可行（UNet 固定 64），改用梯度检查点。
- inpaint 模型的训练分布是"黑色洞"，而我们给的是"正常背景 + 零填充洞"；前者更匹配。当前实现是 `normal_latent * (1 - mask)`，即洞内为零，这正是 SD inpainting 的训练约定。
- 若 SD2-inpaint 只在真实感上提升、检测不升，则不作为主骨干，只作为消融/讨论。

## 6. 与论文叙事的关系

如果 B1/B2 有稳定收益，论文可以把"backbone-task alignment"写成一个小贡献：**inpainting backbone 与 mask-conditioned synthesis 的结构匹配**。如果没有，就在论文中明确"backbone 不是瓶颈，形状/可见性才是"，这本身也是有价值的结论，并能强化主贡献的说服力。
