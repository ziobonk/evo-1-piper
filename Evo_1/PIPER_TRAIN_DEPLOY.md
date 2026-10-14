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
  --run_name piper_stage1 --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --action_head flowmatching --finetune_action_head \
  --per_action_dim 24 --state_dim 24 --horizon 50 \
  --image_size 448 --batch_size 8 --num_workers 2 --prefetch_factor 2 \
  --lr 1e-5 --max_steps 10000 --warmup_steps 500 \
  --ckpt_interval 1000 --log_interval 10
```

先前作业在第 0 步后崩溃且还没有保存 checkpoint，因此第一阶段从头运行。第 10,000 步应生成 `piper_stage1/step_10000` 和 `step_final`。请先确认文件存在，再启动第二阶段：

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

训练中断后，选择**同阶段**最后一个完整的数值目录（例如 `step_2000`），在原命令中添加 `--resume --resume_path /.../step_2000`；第二阶段续训时删除 `--resume_pretrain`，并保持 `--max_steps 3000` 作为累计目标步数。不要把第二阶段的续训 checkpoint 指向第一阶段，也不要用 `step_final` 做常规续训。`step_best` 仅按单个训练 batch 的 loss 选出，不是验证集最优；优先比较固定间隔的 checkpoint 在独立演示和机器人上的表现。当前训练脚本没有验证集评估，不能仅凭训练 loss 判定可部署。

缓存里记录了 `action_horizon`。两阶段保持 `--horizon 50` 和同一个 `--cache_dir`；若要试 10 帧窗口，请换一个缓存目录以重新导出数据。已导出的 50 帧缓存可以继续使用。

## 3. 推理服务

选择实际生成的 checkpoint 目录，其中须有 `config.json`、`norm_stats.json` 与 DeepSpeed 的 `mp_rank_00_model_states.pt` 或普通训练的 `checkpoint.pt`。

```bash
cd Evo_1
python scripts/Evo1_server.py \
  --ckpt_dir ./checkpoints/piper_stage1/step_final \
  --port 9000 \
  --arm_key piper_eef_absolute \
  --dataset_key piper_sdk_raw_eef_next_pose
```

如果 GPU 服务不在机器人电脑上，把客户端的 `--server` 设成 GPU 机器的实际 IP。先确认网络可达、两台机器使用同一套相机视角与 TCP/夹爪标定。

## 4. Piper 客户端

在控制 Piper 的电脑上安装 `piper_sdk`、`scipy`、`opencv-python`、`websockets`，并激活 `can0`。脚本默认只读取机械臂反馈并打印预测动作；明确加 `--execute` 才会使能与下发。相机参数是 OpenCV 设备号或 V4L 路径，顺序必须是训练数据的外部相机、腕部相机。

```bash
cd /home/rhr/openpi/piper_sdk && python -m pip install -e .
cd /home/rhr/litevla/Evo-1/Evo_1
python scripts/Evo1_client_piper.py \
  --dataset ./piper_data \
  --server ws://127.0.0.1:9000 \
  --base-camera /dev/video10 --wrist-camera /dev/video4 \
  --cycles 20 --steps-per-chunk 3
```

先核对打印的目标位姿与当前反馈、摄像头画面、夹爪开合方向和可达范围。确认后再使用相同命令加 `--execute`。客户端按数据集 `tcp_offset_m` 转换 TCP 与法兰；本数据的偏移是零。夹爪宽度依 `gripper_mapping.robot_open_width_mm=70` 映射到 Piper 原生 0..70000。它每次只执行动作块前 3 帧，按 20 Hz 下发，并在每帧前检查位移、旋转、位置范围及夹爪范围。

当前客户端只支持 Piper 原生夹爪与 OpenCV 可打开的相机。若实机使用达妙夹爪或 RealSense 专用采集管线，须把相应驱动接入 `Evo1_client_piper.py`，保证反馈与命令都仍是数据集定义的 0..70 mm 语义。实机执行、相机标定与 GPU 训练结果在当前机器上均未验证。
