# 用 Piper SDK 原始 EEF 数据训练并部署 Evo-1

本说明针对 `/home/rhr/openpi/local/datasets/piper_sdk_raw_eef`。该数据为 LeRobot v2.1，41 个 episode、14,853 帧、20 Hz；`state` 和 `actions` 在同一帧完全相同，都是机器人基座下的绝对 TCP 位姿 `[x,y,z,rotvec_x,rotvec_y,rotvec_z,夹爪宽度 mm]`。因此不能直接把 `actions` 当作下一步命令，也不能对同一帧的 `actions-state` 求差训练。这里让第 `t` 帧预测第 `t+1` 帧的**绝对** TCP 位姿。模型输出也按绝对目标执行，不累加。

**数据要求说明：**论文不要求先用 OXE、DROID 或 LIBERO 等额外机器人数据预训练。它使用已经预训练的 InternVL3-1B 视觉语言骨干；第一阶段用任务演示训练动作专家和整合模块，第二阶段从第一阶段 checkpoint 继续，在同一任务数据上解冻 VLM。README 将第一阶段称为 “pretraining”，其中的 LIBERO 下载仅是示例。当前 Piper 数据可以用于这两个阶段，但只有单任务 41 条演示，不能据此期待论文在更多演示、更广任务分布上报告的成功率。

## 1. 环境与数据准备

在仓库根目录安装训练依赖；Piper 客户端所在环境另需 `scipy` 和本机 Piper SDK。

```bash
python -m venv .venv
.venv/bin/pip install -r Evo_1/requirements.txt
.venv/bin/pip install scipy
```

准备只读数据视图。脚本复制少量 metadata、计算对应的归一化统计，并把 `data/` 链接到原数据；不会修改 `/home/rhr/openpi` 下的文件。

```bash
.venv/bin/python Evo_1/scripts/prepare_piper_dataset.py \
  --source /home/rhr/openpi/local/datasets/piper_sdk_raw_eef \
  --output Evo_1/piper_data
```

检查 [piper_eef_config.yaml](dataset/piper_eef_config.yaml) 里的 `path` 是否是当前仓库的绝对路径。训练读取 Parquet 中的内嵌 JPEG，`image` 和 `wrist_image` 分别是前两路，第三路图像置零并屏蔽。**不要再次运行原仓库的 `compute_normstats`：准备脚本已经生成与下一帧目标匹配的 `meta/stats.json`。**

## 2. 正式训练（单卡 AutoDL 起点）

先把本仓库最新提交通过 Git 同步到 AutoDL：在本机仓库根目录执行 `git push origin main`，再在 AutoDL 仓库根目录执行 `git pull --ff-only`。`524a0bb` 修复了 `--disable_wandb` 下第 0 步之后调用未初始化的 SwanLab 所导致的崩溃；服务器上的旧代码须更新。以下命令从 AutoDL 的 `~/autodl-tmp/evo-1-piper/Evo_1` 运行，假设已配置好环境和 `accelerate config`，并已把 `dataset/piper_eef_config.yaml` 中的 `path` 改为服务器上**实际存在**的 `piper_data` 绝对路径。第一步先检查：

```bash
cd ~/autodl-tmp/evo-1-piper/Evo_1
git log -1 --oneline
python - <<'PY'
from pathlib import Path
import yaml
config = yaml.safe_load(Path('dataset/piper_eef_config.yaml').read_text())
path = Path(config['data_groups']['piper_eef_absolute']['piper_sdk_raw_eef_next_pose']['path'])
print(path, 'exists:', path.is_dir(), 'stats:', (path / 'meta/stats.json').is_file())
PY
```

本数据只有 41 条演示、14,853 帧。下列 10,000 步第一阶段和 3,000 步第二阶段只是可执行的初始方案，需要结合独立演示或实机成功率挑选 checkpoint；不要直接套用论文在 Meta-World 上的 10k + 65k 步。第一阶段冻结 VLM，只训练整合模块与动作专家。沿用已在 AutoDL 成功完成第 0 步的 `horizon 50`、batch 8、ZeRO 2 和缓存目录；50 帧在 20 Hz 下覆盖 2.5 秒。实际状态和动作均为 7 维，训练器仍需填充为 24 维。

```bash
mkdir -p /root/autodl-fs/evo/checkpoints
accelerate launch --num_processes 1 --num_machines 1 \
  --deepspeed_config_file ds_config.json scripts/train.py \
  --dataset_config_path dataset/piper_eef_config.yaml \
  --cache_dir ./piper_cache \
  --save_dir /root/autodl-fs/evo/checkpoints/piper_stage1 \
  --run_name piper_stage1 \
  --vlm_name OpenGVLab/InternVL3-1B \
  --action_head flowmatching --finetune_action_head \
  --per_action_dim 24 --state_dim 24 --horizon 50 \
  --image_size 448 --batch_size 8 --num_workers 2 --prefetch_factor 2 \
  --lr 1e-5 --max_steps 10000 --warmup_steps 500 \
  --ckpt_interval 1000 --log_interval 10
```

先前作业在第 0 步后崩溃且还没有保存 checkpoint，因此第一阶段从头运行。第 10,000 步应生成 `piper_stage1/step_10000`；新版训练脚本在周期 checkpoint 已保存最终步时不再重复写 `step_final`。请先确认文件存在，再启动第二阶段：

```bash
ls /root/autodl-fs/evo/checkpoints/piper_stage1/step_10000/config.json
accelerate launch --num_processes 1 --num_machines 1 \
  --deepspeed_config_file ds_config.json scripts/train.py \
  --dataset_config_path dataset/piper_eef_config.yaml \
  --cache_dir ./piper_cache \
  --save_dir /root/autodl-fs/evo/checkpoints/piper_stage2 \
  --run_name piper_stage2 --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --action_head flowmatching --finetune_vlm --finetune_action_head \
  --per_action_dim 24 --state_dim 24 --horizon 50 \
  --image_size 448 --batch_size 8 --num_workers 2 --prefetch_factor 2 \
  --lr 1e-6 --max_steps 3000 --warmup_steps 200 \
  --ckpt_interval 500 --log_interval 10 \
  --resume --resume_pretrain \
  --resume_path /root/autodl-fs/evo/checkpoints/piper_stage1/step_10000
```

`--resume_pretrain` 只用于**第一阶段转第二阶段**：它加载模型权重，不恢复优化器，并把步数重置为 0。第二阶段解冻整个 VLM，显存需求会明显高于第一阶段；如果 batch 8 OOM，可以将 `--batch_size` 改为 2，同时新建一份 DeepSpeed 配置，将 `train_micro_batch_size_per_gpu` 改为 2，并按显存情况启用 CPU optimizer offload。不能只改命令行 batch 而保留 `ds_config.json` 中固定的 8。较小 batch 会改变每步看到的样本数，需据此重新决定总步数。

训练中断后，选择**同阶段**最后一个完整的数值目录（例如 `step_2000`），在原命令中添加 `--resume --resume_path /.../step_2000`；第二阶段续训时删除 `--resume_pretrain`，并保持 `--max_steps 3000` 作为累计目标步数。不要把第二阶段的续训 checkpoint 指向第一阶段，也不要用旧版 `step_final` 做常规续训。新版 `step_best` 每逢 `--ckpt_interval` 比较前一段训练的平均 loss，从已保存的数值 checkpoint 建立硬链接，因而最多每个周期更新一次，也不会额外复制大文件。它仍只是训练集指标，不是验证集最优；优先比较固定间隔的 checkpoint 在独立演示和机器人上的表现。当前训练脚本没有验证集评估，不能仅凭训练 loss 判定可部署。

缓存里记录了 `action_horizon`。两阶段保持 `--horizon 50` 和同一个 `--cache_dir`；若要试 10 帧窗口，请换一个缓存目录以重新导出数据。已导出的 50 帧缓存可以继续使用。

### 磁盘写满后的续训

DeepSpeed 写入中途失败的 `step_*` 目录可能不完整。先在 AutoDL 检查空间和 checkpoint 大小，不要从报错时正在写入的目录恢复。以第二阶段在 `step_11000` 写入失败为例，最近完整的数值目录通常是 `step_10500`，但须按实际文件确认：

```bash
CKPT=/root/autodl-fs/evo/checkpoints/piper_stage2
df -h "$CKPT"
df -i "$CKPT"
du -sh "$CKPT"/step_* | sort -h
ls -lh "$CKPT/step_10500"
```

可续训的单卡 DeepSpeed 目录至少应有非空的 `mp_rank_00_model_states.pt`、`*_optim_states.pt`、`checkpoint.json`、`config.json` 和 `norm_stats.json`。BF16 训练的优化器文件名可能以 `bf16_zero_pp_rank_0_` 开头。`checkpoint.json` 在模型和优化器保存成功后才写入。若 `step_11000` 缺少它，可删除这个不完整目录；先保留 `step_10500`，再把更早且确认不需要的 checkpoint 移到其他磁盘或逐个删除，至少腾出**一个完整 checkpoint 的大小**，还要留出日志和缓存余量。旧版 `step_best` 来自单 batch loss，内部步数为字符串 `best`，不适合直接作为普通续训源；如需比较推理效果，可单独保留它。新版 `step_best` 来自数值 checkpoint 的硬链接，源目录后来被清理时它仍可读取。

```bash
test -s "$CKPT/step_10500/checkpoint.json" && \
test -s "$CKPT/step_10500/config.json" && \
test -s "$CKPT/step_10500/norm_stats.json" && \
test -s "$CKPT/step_10500/mp_rank_00_model_states.pt" && \
test -n "$(find "$CKPT/step_10500" -maxdepth 1 -type f -name '*_optim_states.pt' -size +0c -print -quit)"
if test -d "$CKPT/step_11000" && ! test -f "$CKPT/step_11000/checkpoint.json"; then
  rm -rf -- "$CKPT/step_11000"
fi
```

然后沿用**原第二阶段命令**及原 `--max_steps` 总步数，只把第一阶段转第二阶段时的 `--resume_path` 改为完整的 `"$CKPT/step_10500"`，保留 `--resume`，并**删除 `--resume_pretrain`**；否则步数会被重置为 0。建议加 `--keep_last_checkpoints 2`：新数值 checkpoint 完整保存后，只保留最近两个数值目录；`step_best` 的硬链接会保留其模型和优化器内容。想保留 best 时**不要加** `--disable_best_checkpoint`；若完全不需要 best，才加该参数。`--ckpt_interval 500` 表示每 500 步最多更新一次 best；若剩余步数不多，可适当增大间隔。新版脚本也会避免在最终步已保存周期 checkpoint 时重复写 `step_final`。下一次保存仍需足够空间。不要把 `--max_steps` 写成“剩余步数”，也不要删正在作为 `--resume_path` 的目录。

如果恢复后几乎每步都打印 `start to save best checkpoint`，说明仍在运行旧版训练器。先中断它，确认一个完整的数值 checkpoint，同步新版代码后重启；只有修改本地文件而不重启 AutoDL 进程不会生效。旧 checkpoint 把最后一个 batch 的 loss 写入 `best_loss`，所以恢复初期容易反复刷新“best”；新版会将不兼容的旧指标重置，并按周期平均 loss 重新选择。`step_best` 仍只基于训练数据，不能代替验证集选择模型。

## 3. 推理服务

选择实际生成的 checkpoint 目录，其中须有 `config.json`、`norm_stats.json` 与 DeepSpeed 的 `mp_rank_00_model_states.pt` 或普通训练的 `checkpoint.pt`。

推理服务从 InternVL3 配置创建模型结构，再严格加载 EVO checkpoint 中的完整权重（含 VLM），不再先下载原始 InternVL3 权重。首次运行仍需要下载或预先缓存 `OpenGVLab/InternVL3-1B` 的配置、tokenizer 和自定义 Python 模型代码；已有这些缓存时可以加 `HF_HUB_OFFLINE=1` 启动。若旧版服务报 `pytorch_model.bin` 不存在并提示 `from_tf=True`，请更新代码后重试，不要改成加载 TensorFlow 权重。官方 InternVL3-1B 使用 `model.safetensors`，镜像访问异常可能触发这个误导提示。

```bash
cd ~/autodl-tmp/evo-1-piper/Evo_1
python scripts/Evo1_server.py \
  --ckpt_dir /root/autodl-fs/evo/checkpoints/piper_stage1/step_10000 \
  --port 9000 \
  --arm_key piper_eef_absolute \
  --dataset_key piper_sdk_raw_eef_next_pose
```

如果第二阶段 checkpoint 的离线验证更好，可将 `--ckpt_dir` 换成其数值 step 目录。如果 GPU 服务不在机器人电脑上，把客户端的 `--server` 设成 GPU 机器的实际 IP。先确认网络可达、两台机器使用同一套相机视角与 TCP/夹爪标定。

服务器会单独记录模型推理用时，默认保存到 `logs/evo1_server_inference_<时间戳>.jsonl`；服务器命令可追加 `--log-path logs/server_inference.jsonl` 指定追加写入的文件。每个请求记录 `model_inference_ms`、`chunk_length`、状态、错误、`request_id` 和启动时的 `run_id`，每条即时 flush。`model_inference_ms` 只测量 `model.run_inference()` 调用：开始前同步 CUDA 排除之前尚未完成的工作，结束后同步 CUDA 等待模型计算完成，再读高精度计时器。它包含模型调用内部的图像处理、tokenization 和 CPU 调度，不等同于 GPU kernel 时间之和；不包含网络传输、JSON 编解码、调用外的图像预处理与上传、状态归一化、动作反归一化和输出下载。第一条请求可能包含模型首次运行的初始化开销。旧客户端仍接收原有动作数组；新版 Piper 客户端通过 `return_timing=true` 请求带 `actions`、`timings` 和匹配 `request_id` 的响应。需要两端分阶段计时时，请重启服务器和客户端；新版客户端也兼容旧服务器返回的数组，但服务器阶段统计为 null。

服务器正常退出、报错或 Ctrl+C 退出时，会在 `session_end` 中写入并打印本次启动的平均模型推理用时 `mean_model_inference_ms`、计入均值的请求数 `inference_count`、总推理用时 `total_model_inference_ms`，以及总请求数、失败数和中断数。均值只统计成功请求，包含首次推理，不统计失败或中断的请求，也不混入同一文件中以前启动的记录；没有成功推理时均值为 null。客户端断开连接不会结束服务器的统计，本次启动期间的多个连接共用统计。

## 4. Piper 客户端

在控制 Piper 的电脑上安装 `piper_sdk`、`scipy`、`opencv-python`、`websockets`、`pyrealsense2`（`python -m pip install pyrealsense2`），并激活 `can0`。脚本默认只读取机械臂反馈并打印预测动作；明确加 `--execute` 才会使能与下发。两台 D435i 以序列号指定，全局相机对应数据集 `image`，手腕相机对应 `wrist_image`。只使用彩色图像；采集分辨率取数据集 metadata 中的 1280×720 和 640×480，默认请求 30 FPS，模型输入统一缩放到 448×448。请先查看实际序列号和相机视角：

```bash
cd /home/rhr/litevla/Evo-1/Evo_1
python scripts/Evo1_client_piper.py --list-cameras
```

```bash
cd /home/rhr/openpi/piper_sdk && python -m pip install -e .
cd /home/rhr/litevla/Evo-1/Evo_1
python scripts/Evo1_client_piper.py \
  --dataset ./piper_data \
  --server ws://localhost:6006 \
  --base-serial 112222070761 --wrist-serial 231122071797 \
  --camera-fps 30 --cycles 20 --steps-per-chunk 3 --control-hz 100
```

先核对打印的目标位姿与当前反馈、摄像头视角、夹爪开合方向和可达范围。确认后再使用相同命令加 `--execute`。客户端按数据集 `tcp_offset_m` 转换 TCP 与法兰；本数据的偏移是零。夹爪宽度依 `gripper_mapping.robot_open_width_mm=70` 映射到 Piper 原生 0..70000。每个预测块只使用前 3 个关键帧，每个关键帧占数据集的 1/20 秒；100 Hz 控制时在两帧之间发 5 条插值命令。位置和夹爪线性插值，姿态使用球面插值。原始关键帧仍须通过相对反馈的位移、旋转和工作空间检查，插值不会放宽限制。推理期间机器人保持上一目标，下一块从新的实测反馈开始。

Piper 客户端现在默认使用 `--image-transport binary`：只发送两路缩放后的 `448×448` BGR uint8 原始图像及小型 JSON 元数据头，图像无损，不转像素列表、不做 JPEG 编码。服务器根据 `[1,1,0]` mask 补第三路零图，保持原有三路模型输入，并批量上传 uint8 图像到 GPU，再转换为 RGB 浮点张量；已是 `448×448` 时跳过重复缩放。两端 WebSocket 默认关闭压缩，适合本地推理。请重启 server 和 client 后使用；连接旧版服务器时可加 `--image-transport json` 回退。新版服务器继续接受旧 JSON 客户端，离线评估脚本的旧格式仍可用。

推理 prompt 默认读取 `piper_data/meta/tasks.jsonl` 的第一条 `task`。可在客户端命令后追加 `--prompt "<control mode> end effector <control mode> pick up the cube and drop it into the box"` 覆盖，本次运行每轮都会发送该文本；启动时打印 `Task prompt`，并在 JSONL 的 `session_start.prompt` 中记录。这里的控制模式前缀与当前训练文本一致；无需修改数据集或重启服务器，只需重启客户端。当前数据集是单任务数据，替换任务文本不代表模型已经学会对应新任务。

`--cycles 20` 会在 `cycle=0` 至 `cycle=19` 完成后正常退出；希望持续推理时改为 `--cycles 0`，用 Ctrl+C 停止。正常结束、异常退出或 Ctrl+C 时，客户端只断开 CAN 通信并释放相机，不再调用 `DisableArm()`，也不会发送新的运动目标或夹爪命令；退出后不主动改变机械臂使能状态。

客户端默认将推理日志逐行保存到 `logs/piper_inference_<时间戳>.jsonl`，启动时打印绝对路径。可加 `--log-path logs/piper_run.jsonl` 指定文件；已有文件会追加，每次启动通过 `run_id` 区分，时间戳采用 UTC+8。日志包含 `session_start`、每轮 `cycle` 和退出时的 `session_end`，每条写入后立即 flush；报错或 Ctrl+C 也会保留当前轮记录和已下发计数。每轮还保存请求时的 7 维反馈、返回动作块的前 7 维、裁剪记录、通过检查的目标、拒绝原因和状态，不保存图像。

| 每轮字段 | 含义 |
| --- | --- |
| `inference_latency_ms` | 客户端开始发送请求至收到完整响应的耗时，包含上传、网络、服务端排队和处理、响应下载；不是纯 GPU 计算时间。未收到完整响应时为 null |
| `request_preparation_ms` | 图像缩放、请求构造和编码耗时，不包含在上述延迟中；binary 模式只序列化小型元数据头，并打包原始图像字节 |
| `chunk_length` | 模型实际返回的动作块长度，如 50 |
| `requested_steps` | 每块计划执行的关键帧数，即 `--steps-per-chunk` |
| `processed_steps` | 完成检查和插值的关键帧数，试运行也计入 |
| `executed_steps` | 所有插值控制点均成功下发的关键帧数；试运行为 0 |
| `sent_control_steps` | 成功完成一次 `EndPoseCtrl` 和 `GripperCtrl` 调用的插值控制点数；试运行为 0 |
| `execution_latency_ms` / `cycle_latency_ms` | 本轮处理或执行动作阶段 / 整轮耗时 |

例如返回 50 步、选前 3 步、20 Hz 数据配合 100 Hz 控制时，一轮正常执行日志为 `chunk_length=50`、`executed_steps=3`、`sent_control_steps=15`。执行计数表示 SDK 命令调用成功，不表示机械臂已经到达目标；中途拒绝或中断时只累计此前完成的部分。退出记录汇总总轮数、已下发关键帧数及插值控制点数。

客户端和服务器现在还各自保存一份独立延迟统计 JSON，默认文件名为各自 JSONL 的同名 `.latency.json`。可在客户端原命令末尾追加 `--latency-json logs/piper_latency.json`；这一份 JSON 会同时包含客户端和服务器分阶段统计。服务器可另加 `--latency-json logs/server_latency.json` 指定其统计文件，请勿让两个进程写同一个文件。每轮完成后更新，退出时再次保存；文件是完整 JSON，采用临时文件替换，便于 IDE 直接查看。每项统计包含 `count`、`mean_ms`、`min_ms`、`max_ms`、`total_ms`，未测量的项计数为 0、均值为 null；包含首次请求，只汇总当前启动的数据。

| 延迟统计字段 | 范围 |
| --- | --- |
| `capture_ms` / `feedback_read_ms` | 客户端采集两路图像 / 读取机械臂反馈 |
| `payload_build_ms` / `request_encode_ms` | 客户端缩放图像、构造请求 / 编码请求；binary 模式无像素列表，只打包元数据头及原始字节 |
| `response_decode_ms` | 客户端解析响应 JSON 和动作数组 |
| `client_data_processing_ms` | 客户端请求构造、序列化、响应解析的合计 |
| `server_json_decode_ms` | 服务器解析 JSON；binary 模式只解析小型元数据头 |
| `server_unpack_ms` | binary 请求校验和创建图像 NumPy 视图的耗时；JSON 模式为 0 |
| `server_preprocess_ms` | 服务器图像转换、上传 GPU、状态归一化与 mask 准备，含 CUDA 同步等待 |
| `server_model_inference_ms` | 客户端收到的服务器模型推理计时；服务器自己的统计中名为 `model_inference_ms` |
| `server_postprocess_ms` | 服务器动作反归一化、复制回 CPU 和转为列表 |
| `server_processing_ms` | 服务器从开始解析请求到完成动作后处理的总时间，涵盖上述服务器阶段及调用间开销 |
| `server_response_encode_ms` | 服务器响应 JSON 序列化，仅在服务器统计中记录 |
| `communication_and_other_ms` | 客户端请求往返耗时减去同一次请求的 `server_processing_ms`；包含传输、WebSocket 压缩/解压、响应序列化、日志写入及调度等，不是纯网络延迟 |

客户端延迟统计计入成功解码的动作响应，即使该动作随后被范围检查拒绝；没有收到动作响应的中断轮次不计入。服务器延迟统计仅计入成功推理请求。两边用请求 ID 匹配数据，无需两台机器时钟同步。模型推理计时和分阶段统计含 CUDA 同步，因此实际阶段边界清晰；各项总时间与其子项有重叠，不应把所有字段相加。

每轮 JSONL 还记录 `image_transport` 和 `request_bytes`，启动记录包含压缩设置。可用 `python scripts/benchmark_inference_transport.py --dataset ./piper_data --iterations 5` 比较旧 JSON 路径与新 binary 路径，结果保存到 `logs/inference_transport_benchmark.json`。该基准使用数据集真实图像，验证 CPU 模型输入逐元素相同，仅测请求构造、编解码和 CPU 预处理，不测网络、压缩、GPU 上传或模型推理；实测收益以重启两端后的延迟统计为准。

Flow matching 输出没有硬性限制，反归一化也不会裁剪，因此闭合附近可能出现负值。客户端在试运行和执行模式下都会将有限的夹爪预测宽度裁剪到 `[0, robot_open_width_mm]`（本数据为 0～70 mm），再插值和下发；例如 `-0.28 mm` 裁剪为 `0 mm`，并打印 `gripper clipped` 及裁剪前后的值。NaN/Inf 仍拒绝，位置、姿态和步长检查保持原有阈值。默认试运行遇到被拒绝的关键帧会打印 `rejected`、跳过该动作块剩余关键帧并继续采样，`--execute` 则立即报错停止。`Target step too large: 0.08 m` 表示位移超过默认 0.02 m 限制，不会因夹爪裁剪而通过。

若仍使用 V4L/OpenCV 相机，可加 `--camera-backend opencv --base-camera /dev/video10 --wrist-camera /dev/video4`。若 D435i 不支持所请求的彩色流规格，先用 `realsense-viewer` 查看两台相机的可用 profile，再调整 `--camera-fps`；切换采集分辨率须注意与训练数据视野一致。当前客户端只支持 Piper 原生夹爪；若实机使用达妙夹爪，须接入对应驱动并保持数据集定义的 0..70 mm 宽度语义。实机执行与相机标定在当前机器上未验证。

## 5. 使用数据集离线验证并可视化

先在 GPU 机器上启动第 3 节的推理服务，再运行下列脚本。它读取数据集 Parquet 中的两路 JPEG、7 维状态和任务文本，发送与实机客户端相同格式的请求；真值采用训练时的“下一帧绝对 TCP 位姿”定义。`--horizon` 须与 checkpoint 一致，当前训练为 50。它不需要连接 Piper 或 D435i。

```bash
cd ~/autodl-tmp/evo-1-piper/Evo_1
python scripts/eval_piper_offline.py \
  --dataset ./piper_data --server ws://127.0.0.1:9000 \
  --episodes 0,10,20,30,40 --samples-per-episode 3 \
  --horizon 50 --output ./piper_offline_eval
```

输出目录包含 `summary.json`（首 1 步、首 3 步和全 50 步的平均位置误差 mm、姿态误差度、夹爪误差 mm）、`errors.csv`（逐预测步误差）和每个样本的 PNG 对比图。图中同时显示两路输入图像、真实与预测 TCP 位置、夹爪宽度、位置及姿态误差。长动作块的后几步是**开环预测**，通常比第 1～3 步更难；本客户端只执行前 3 步。现有 41 条数据均参与训练，直接回放它们只是训练集诊断，不能代表未见场景或实机成功率。需要可信的泛化评估时，应先按 episode 留出数据并重新训练。
