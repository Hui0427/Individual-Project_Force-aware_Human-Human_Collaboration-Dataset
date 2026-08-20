# 在 HPC 上跑 HOI 流水线

标定(mesh→rigid body)已经做完,**HPC 上不需要跑标定**。需要上传的只有:

```
scripts/                      # 全部脚本
outputs/mesh_offsets.json     # 标定结果(纯数字,与机器无关)
takes.yaml                    # take 配置(把路径改成集群上的)
requirements.txt
```

## 一次性环境

```bash
module load python            # 或集群对应的方式
python -m venv ~/venvs/hoi
source ~/venvs/hoi/bin/activate
pip install -r requirements.txt
```

## 用法:一条命令跑一条 take

所有参数都写在 `takes.yaml` 里,命令行不用填路径:

```bash
python scripts/run_take.py --config takes.yaml --take drill_shot005     # 单条
python scripts/run_take.py --config takes.yaml --all                    # 全部
python scripts/run_take.py --config takes.yaml --all --stages refine,export   # 只重跑某几步
```

四个阶段:`extract`(提取位姿/手关节/局部旋转)→ `annotate`(距离+contact+图)
→ `refine`(手部姿态优化)→ `export`(Unity .anim + 回放 json)。

**已存在的产出会自动跳过**,失败重跑不用从头来;要强制重算加 `--force`。
某条 take 失败不会中断其余,最后统一报告。每条 take 有独立 `run.log`。

## takes.yaml 要改什么

只改路径两项,其余照抄:

```yaml
"drill_shot005": {
  "object": "drill",                      # mesh_offsets.json 里的物体名
  "take_key": "shot_005/Drill.skel",      # mesh_offsets.json 里的 key,不是路径!
  "object_fbx": "Drill",                  # take 文件夹里物体 fbx 的文件名
  "data_dir": "/集群上/shot_005",          # ← 改这里
  "mesh": "/集群上/035_power_drill/google_16k/textured.obj",   # ← 和这里
  "refine_persons": ["person1"],          # 要优化谁的手(省略=全部)
  "frames": [0, -1]                       # [0,-1] = 整条 take;也可以给 [150,270]
}
```

`defaults` 段里的全局项:

| 项 | 说明 |
|---|---|
| `sampled` | **删掉这一行 = 精确模式(完整版)**,单 take 10~20 分钟。填 500000 是快速模式(~10 秒,实测误差 0.02±0.03mm) |
| `contact_thr_mm` | contact 阈值,默认 15(关节中心 + 指半径 + 标定误差底,依据见 README) |
| `iters` | 优化迭代次数,默认 300;整条 take 建议 400 |
| `mode` | `anatomical`(默认,关节只沿自身铰链轴动)/ `free`(无约束,会产生不像人手的姿态,仅供对比) |

## SLURM job array

```bash
#!/bin/bash
#SBATCH --job-name=hoi
#SBATCH --array=0-4            # take 数量 - 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00        # 精确模式 + 整条 take 优化,留足余量

source ~/venvs/hoi/bin/activate
cd ~/core4d_refine_project

NAMES=(drill_shot005 drill_shot007 crate_shot012 hammer_0728shot007 spray_0728shot020)
python scripts/run_take.py --config takes.yaml --take ${NAMES[$SLURM_ARRAY_TASK_ID]}
```

`torch` 只用 CPU(问题规模小,GPU 没有收益),所以不用申请 GPU 节点。

## 坑位

- **`take_key` 必须和 `mesh_offsets.json` 里的完全一致**,别改成 HPC 路径
- **新 take 要先在本地做标定**:把该 take 的 `.skel` 加进 `fit_captury_motive_alignment.py`
  的 `SKEL_FILES`,重跑它和 `build_mesh_offsets.py`,再把新的 mesh_offsets.json 传上去。
  那一步依赖本地的 `.motive` 文件,别在 HPC 上折腾
- person 命名不统一(有的 take 是 `p1`/`p2`,新的一批是 `person`/`person2`/`person3`),
  按实际文件名填 `persons`
- `rtree` 需要 libspatialindex,pip 的 wheel 一般自带;装不上就 `conda install rtree`

## 产出

```
outputs/takes/<take名>/
├── distances.csv              逐帧逐关节 signed distance + contact 标签
├── summary.json               contact 段落、穿透帧、异常帧、阈值敏感度
├── fig1~3.png                 距离曲线 / contact 时间线 / 最近帧 3D 检查
├── run.log
└── refine_<person>/
    ├── hand_params_orig.npz   原始骨架参数(每帧每关节 local quat)
    ├── hand_params_refined.npz 优化后 + delta
    ├── skeleton_meta.json     层级/骨长/FPS/权重/可信度统计/改动量
    ├── replayer_<person>.json → Unity HandPoseReplayer.cs 用
    └── refined_<person>.anim  → Unity AnimationClip
```
