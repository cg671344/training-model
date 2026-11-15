# 大模型后训练实战全流程（`NEXUS-DSL v1`）

[English Version](./README.md) | 简体中文

本项目在本地 **NVIDIA GTX 1650（4 GB 显存）** 上，演示了如何基于原始基座模型（Base Foundation Model：`Qwen/Qwen2.5-0.5B`）完成端到端的 **后训练（Post-Training）** 全流程。

## 为什么选择这个任务？（“训练前 vs. 训练后”能力对比）

原始基座模型在互联网预训练语料中**从未见过**我们自定义的智能家居编译协议（`NEXUS-DSL v1`）：
```text
@NEXUS_V1
CMD lum.kitchen.main := ON;
CMD hvac.kitchen.temp := 22C;
@END
```
- **后训练前（Base 模型）：** 语法有效率 **`0%`**，精确匹配率 **`0%`**（基座模型只会续写提示词或输出无关乱码）。
- **后训练后（LoRA SFT）：** 能够稳定解析单指令、多意图复合指令，以及严格按协议拒答超纲问题（`ERR_SCOPE := OUT_OF_DOMAIN;`），语法有效率达到 **90.0%**，精确匹配率达到 **85.0%**。

---

## 1. 量化评测结果：训练前 vs. 训练后

| 评测指标（20 条未见过的保留测试集） | 训练前（基座模型 `Qwen/Qwen2.5-0.5B`） | 训练后（`Base + LoRA SFT`） | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **`NEXUS-DSL v1` 语法有效率** | `0 / 20` (**0.0%**) | `18 / 20` (**90.0%**) | **+90.0%** |
| **指令精确匹配率（Exact Match）** | `0 / 20` (**0.0%**) | `17 / 20` (**85.0%**) | **+85.0%** |
| **超纲问题拒答准确率** | `0 / 4` (**0.0%**) | `4 / 4` (**100.0%**) | **+100.0%** |
| **多意图组合指令准确率** | `0 / 4` (**0.0%**) | `3 / 4` (**75.0%**) | **+75.0%** |

### 训练 Loss 收敛过程（[`results/training_log.json`](./results/training_log.json)）
- **基座模型：** `Qwen/Qwen2.5-0.5B`
- **硬件环境：** NVIDIA GeForce GTX 1650（4 GB 显存，`fp16` 混合精度）
- **可训练参数量：** `8,798,208` / `502,830,976`（LoRA `r=16, alpha=32`，仅训练 **1.75%** 参数）
- **训练前初始验证集 Loss：** `3.7875`
- **Epoch 1：** Train Loss `1.1684` | Val Loss `0.4108`
- **Epoch 2：** Train Loss `0.3576` | Val Loss `0.3772`
- **Epoch 3：** Train Loss `0.3566` | Val Loss `0.3700`
- **Epoch 4：** Train Loss `0.3525` | Val Loss `0.3687`

---

## 2. 训练前后输出逐条对比（[`before_training.json`](./results/before_training.json) vs. [`after_training.json`](./results/after_training.json)）

| 测试类别 | 未见过的用户提问 | 训练前（原始基座模型） | 训练后（LoRA SFT 适配器） |
| :--- | :--- | :--- | :--- |
| **单指令控制** | *"Could you switch on the kitchen lights please?"* | `'Yes, I can switch on the kitchen lights.erable erable Could you switch...'` | `@NEXUS_V1`<br>`CMD lum.kitchen.main := ON;`<br>`@END` |
| **同义词映射** (*office → study*) | *"Set the thermostat in my office to 21 degrees."* | `'Set the thermostat in my office to 21 degrees.})(); Comey Set the...'` | `@NEXUS_V1`<br>`CMD hvac.study.temp := 21C;`<br>`@END` |
| **同义词映射** (*curtains → blinds*) | *"Close the living room curtains."* | `"I'm sorry, but I don't understand the request. Can you please provide..."` | `@NEXUS_V1`<br>`CMD shade.living.blinds := CLOSED;`<br>`@END` |
| **多意图组合** | *"Turn on the lights in the study and set the temperature in the study to 23 degrees."* | `'What is the temperature in the study? ViewById: 1 ViewById: 2...'` | `@NEXUS_V1`<br>`CMD lum.study.main := ON;`<br>`CMD hvac.study.temp := 23C;`<br>`@END` |
| **跨领域组合** | *"Close the blinds in the bedroom and switch off the television in the bedroom."* | `'I want to turn on the lights in the bedroom and turn off the...'` | `@NEXUS_V1`<br>`CMD shade.bedroom.blinds := CLOSED;`<br>`CMD media.bedroom.tv := OFF;`<br>`@END` |
| **超纲拒答测试** | *"How do I install PyTorch on Windows?"* | `'How do I install PyTorch on Windows?forgettable forgettable...'` | `@NEXUS_V1`<br>`ERR_SCOPE := OUT_OF_DOMAIN;`<br>`@END` |
| **超纲拒答测试** | *"Write a haiku about autumn leaves."* | `'Write a haiku about autumn leaves.})(); })(); })(); })();...'` | `@NEXUS_V1`<br>`ERR_SCOPE := OUT_OF_DOMAIN;`<br>`@END` |

---

## 3. 本次实战总结的三个关键经验

1. **仅针对回答部分的 Loss 掩码（Completion-Only Loss Masking，见 [`03_train_sft.py`](./03_train_sft.py)）：**
   在样本 #0（共 107 个 Token）中，前 85 个系统提示词与用户输入 Token 的标签全部被掩码设为 `label = -100`，使 100% 的梯度更新都集中在最后 22 个目标输出 Token（`@NEXUS_V1\nCMD hvac.study.temp := 22C;\n@END`）上。
2. **原始基座模型中未训练的特殊结束符（见 [`04_eval_after.py`](./04_eval_after.py)）：**
   在原始 `Qwen/Qwen2.5-0.5B` 基座模型中，`tokenizer.eos_token` 是 `<|endoftext|>`（ID `151643`），而 ChatML 对话模板使用的是 `<|im_end|>`（ID `151645`）。由于 LoRA 默认冻结 `embed_tokens` 和 `lm_head`，输出头里的 `<|im_end|>` 权重尚未被激活；因此在推理时需要传入 `stop_strings=["@END"]`（或在 `LoraConfig` 中设置 `modules_to_save=["embed_tokens", "lm_head"]`）来保证干净截断。
3. **通过错误分析定位数据分布不均：**
   测试集中有 2 条语法错误（例如第 8 题 *"Arm the security alarm for the entire house"* 被输出为 `CMD sec.house.alarm := ARMED;`，而非标准的 `sec.all.alarm`），原因是训练集中 `all` 区域仅有 6 条样本，而其他具体房间各有约 40 条，导致模型直接照抄了英文名词 `house` 作为区域标识。补充 15~20 条全屋（`all`）指令样本即可直接修复。

---

## 4. 项目文件说明

| 文件 | 阶段 / 角色 | 说明 |
| :--- | :--- | :--- |
| [`dsl_validator.py`](./dsl_validator.py) | 规则验证器 | 纯 Python 编写的 `NEXUS-DSL v1` 确定性语法解析器与语义等价评测器。 |
| [`01_prepare_data.py`](./01_prepare_data.py) | 步骤 1：数据准备 | 生成 [`data/sft_train.jsonl`](./data/sft_train.jsonl)（258 条训练集）、[`data/sft_val.jsonl`](./data/sft_val.jsonl)（30 条验证集）、[`data/test_heldout.jsonl`](./data/test_heldout.jsonl)（20 条保留测试集）和 [`data/dpo_train.jsonl`](./data/dpo_train.jsonl)（60 条 DPO 偏好对）。 |
| [`02_eval_before.py`](./02_eval_before.py) | 步骤 2：基线评测 | 在训练前评测原始基座模型（`Qwen/Qwen2.5-0.5B`），并将结果保存至 [`results/before_training.json`](./results/before_training.json)。 |
| [`03_train_sft.py`](./03_train_sft.py) | 步骤 3：LoRA SFT 训练 | 使用 **Completion-Only Loss Masking** 运行监督微调，将 LoRA 适配器保存至 `checkpoints/nexus-sft-lora`，训练日志保存至 [`results/training_log.json`](./results/training_log.json)。 |
| [`04_eval_after.py`](./04_eval_after.py) | 步骤 4：效果对比评测 | 在同一份保留测试集上评测微调后模型，输出训练前后对比并保存至 [`results/after_training.json`](./results/after_training.json)。 |
| [`05_train_dpo.py`](./05_train_dpo.py) | 步骤 5：DPO 偏好对齐 | 演示第二阶段直接偏好优化（`chosen` vs `rejected`），利用 `model.disable_adapter()` 在 4 GB 显存上零额外基座开销完成 DPO 训练。 |

---

## 快速复现命令

```powershell
# 1. 生成训练、验证、测试与 DPO 数据集
.venv\Scripts\python.exe 01_prepare_data.py

# 2. 评测训练前的原始基座模型
.venv\Scripts\python.exe 02_eval_before.py

# 3. 运行 LoRA 监督微调（SFT）
.venv\Scripts\python.exe 03_train_sft.py

# 4. 评测微调后模型并输出训练前后对比
.venv\Scripts\python.exe 04_eval_after.py

# 5. （可选）运行第二阶段 DPO 偏好对齐
.venv\Scripts\python.exe 05_train_dpo.py
```

---

## 5. 核心原理解析（面向初学者的深度拆解）

### 5.1 为什么只对“回答部分”计算 Loss？（Completion-Only Loss Masking）

在样本 #0（共 107 个 Token）中，前 85 个系统提示词与用户输入 Token 的标签全部被掩码设为 `label = -100`，使 100% 的梯度更新都集中在最后 22 个目标输出 Token（`@NEXUS_V1\nCMD hvac.study.temp := 22C;\n@END`）上。

- **背景知识：** 训练语言模型时，我们会把整段对话拼成 Token ID 序列，同时准备一份标准答案列表 `labels`。PyTorch 的 `CrossEntropyLoss` 规定：**只要 `labels` 中某一位是 `-100`，计算 Loss 时就会自动跳过该位置，不产生梯度。**
- **为什么需要掩码：** 一条样本包含两段：
  - **前 85 个 Token（考题）：** System Prompt + User 提问 + `<|im_start|>assistant`
  - **后 22 个 Token（答案）：** `@NEXUS_V1\nCMD hvac.study.temp := 22C;\n@END<|im_end|>`
- 如果不把前 85 个 Token 设为 `-100`，模型就会被迫把 80% 的算力浪费在“背诵题干”上，就像考试时要求学生连题目一起默写。在 [`03_train_sft.py`](./03_train_sft.py) 的 `CompletionOnlySFTDataset` 中，我们通过：
  ```python
  labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
  ```
  让模型完整阅读前面的上下文，但**只考核并优化最后 22 个答案 Token 的预测准确率**。

### 5.2 为什么原始基座模型写完 `@END` 后不肯停止？（未训练的特殊结束符）

大模型没有“发完消息自动停机”的本能，它必须自己生成一个特殊的**结束符号（EOS Token）**，推理框架（`model.generate`）检测到这个 ID 后才会停止生成。

在原始 `Qwen/Qwen2.5-0.5B` 基座模型中：
- **基座模型预训练用的结束符：** `<|endoftext|>`（ID `151643`），用于分隔互联网网页文本。
- **ChatML 对话模板用的结束符：** `<|im_end|>`（ID `151645`），用于标记一轮助手回答结束。在未经过对话微调的原始基座模型中，`<|im_end|>` 几乎从未被激活。
- **LoRA 的默认冻结机制：** 大模型结构分为入口词表（`embed_tokens`）→ 中间 Transformer 层 → 出口输出头（`lm_head`）。LoRA 默认只微调中间层，**冻结了 `embed_tokens` 和 `lm_head`**。这导致模型虽然在中间层学会了“写完 `@END` 就该结束”，但出口输出头里 `<|im_end|>` 的概率权重依然是冻住的，无法主动吐出它，于是只能继续往下胡言乱语。
- **两种解决方法：**
  1. **推理侧拦截（[`04_eval_after.py`](./04_eval_after.py) 采用）：** 在 `model.generate(...)` 中传入 `stop_strings=["@END"]`，一旦生成 `@END` 立即强制截断。
  2. **训练侧解冻（治本方案）：** 在 `LoraConfig` 中加入 `modules_to_save=["embed_tokens", "lm_head"]`，允许输出头一起更新权重，让模型真正学会自己在结尾吐出 `<|im_end|>`（代价是可训练参数与显存占用会有所增加）。

### 5.3 如何通过错误分析定位数据分布不均？（`sec.house.alarm` vs `sec.all.alarm`）

在 20 道测试题中，第 8 题（*"Arm the security alarm for the entire house."*）和第 12 题（*"Turn off every light in the whole house."*）均出现了同一类错误：模型输出了 `sec.house.alarm` 和 `lum.house.main`，而不是合法的 `sec.all.alarm` 与 `lum.all.main`。

- **根因定位：** 回查 [`01_prepare_data.py`](./01_prepare_data.py) 中的样本分布可以发现，6 个具体房间（`kitchen`、`bedroom`、`living`、`study`、`garage`、`porch`）各有约 **40 条**训练样本，模型被充分训练去学习“房间词 → 区域 ID”的映射；而全屋区域 `all` 在整个训练集中只有 **6 条**样本。
- **模型行为退化：** 因为 `all` 的样本太少，模型在测试集遇到未见过的 `"entire house"` / `"whole house"` 时，无法稳定映射到 `all`，只能退化为**直接照抄英文名词 `house`** 填入区域槽位。
- **工程启示：** 后训练阶段遇到特定场景出错时，**先做错误分析定位数据短板，往往比盲目调学习率或增加 Epoch 更有效**。只需在 [`01_prepare_data.py`](./01_prepare_data.py) 中为 `all` 区域补充 15~20 条多样化表达（如 *"entire house"*、*"whole home"*、*"all rooms"*）并重跑训练即可直接修复。

