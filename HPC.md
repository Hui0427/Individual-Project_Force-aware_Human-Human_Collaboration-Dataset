# 在 HPC 上跑 HOI 距离流水线

脚本已全部参数化,**没有任何硬编码数据路径**(标定脚本除外——标定已完成,不需要在 HPC 重跑)。
需要上传的只有三样:

```
scripts/                      # 五个下游脚本
outputs/mesh_offsets.json     # 标定结果(纯数字,与机器无关)
requirements.txt
```

## 一次性环境

```bash
module load python  # 或集群对应的方式
python -m venv ~/venvs/hoi
source ~/venvs/hoi/bin/activate
pip install -r requirements.txt
```

## 每条 take 三步(路径全是变量,按你的 HPC 数据位置改)

```bash
DATA=/path/to/your/shot_005          # take 文件夹(含 *.fbx)
MESH=/path/to/035_power_drill/google_16k/textured.obj
OUT=~/hoi_out/drill_shot005

python scripts/extract_object_pose.py        $DATA/Drill.fbx   $OUT/drill_pose.csv
python scripts/extract_person_hand_joints.py $DATA/person1.fbx $OUT/person1_hand_joints.csv
python scripts/extract_person_hand_joints.py $DATA/person2.fbx $OUT/person2_hand_joints.csv

python scripts/compute_hand_object_distances.py \
    --object drill --take "shot_005/Drill.skel" \
    --mesh $MESH --object-pose $OUT/drill_pose.csv \
    --hands $OUT/person1_hand_joints.csv $OUT/person2_hand_joints.csv \
    --out-dir $OUT          # 默认精确模式(完整版);加 --sampled 500000 可提速 ~60 倍

python scripts/plot_hoi_results.py --out-dir $OUT --mesh $MESH \
    --object drill --take "shot_005/Drill.skel"
```

注意:
- `--take` 是 `mesh_offsets.json` 里 `takes{}` 的 key,**不是路径**,别改成 HPC 路径
- **默认就是精确模式(完整版)**,单 take 约 10-20 分钟。`--sampled 500000` 是快速模式
  (单 take 约 10 秒),实测 vs 精确解误差 0.02±0.03mm(500 点抽查,最大 0.22mm)
- 单条 take 本地都只要几十秒(提取)+10 秒(距离),HPC 的价值在**批量并行**,不在单条提速

## SLURM 批量模板(job array,每个任务一条 take)

```bash
#!/bin/bash
#SBATCH --job-name=hoi
#SBATCH --array=0-5           # take 数量 - 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00   # 精确模式留足余量

source ~/venvs/hoi/bin/activate
cd ~/core4d_refine_project

# 每行: 物体名  take_key  take数据目录  mesh路径  物体fbx名
TAKES=(
  "drill  shot_005/Drill.skel        /data/.../new_data_0716/shot_005  /data/.../035_power_drill/google_16k/textured.obj  Drill"
  "drill  shot_007/Drill.skel        /data/.../new_data_0716/shot_007  /data/.../035_power_drill/google_16k/textured.obj  Drill"
  "hammer 0728_shot_007/Hammer.skel  /data/.../0728_data/0728_shot_007 /data/.../048_hammer/google_16k/textured.obj       Hammer"
  # ...按 INDEX.md 的 take 表补全
)
read OBJ TAKE DATA MESH FBX <<< "${TAKES[$SLURM_ARRAY_TASK_ID]}"
OUT=~/hoi_out/${OBJ}_$(basename $DATA)
mkdir -p $OUT

python scripts/extract_object_pose.py        $DATA/$FBX.fbx    $OUT/${OBJ}_pose.csv
python scripts/extract_person_hand_joints.py $DATA/person1.fbx $OUT/person1_hand_joints.csv
python scripts/extract_person_hand_joints.py $DATA/person2.fbx $OUT/person2_hand_joints.csv
python scripts/compute_hand_object_distances.py --object $OBJ --take "$TAKE" \
    --mesh $MESH --object-pose $OUT/${OBJ}_pose.csv \
    --hands $OUT/person*.csv --out-dir $OUT
python scripts/plot_hoi_results.py --out-dir $OUT --mesh $MESH \
    --object $OBJ --take "$TAKE" --object-pose-name ${OBJ}_pose.csv
```

坑位提醒:
- **take_key 必须和 `mesh_offsets.json` 里的完全一致**(如 `shot_005/Drill.skel`);
  新 take 要先在本地跑 `fit_captury_motive_alignment.py` 把该 take 的 `.skel` 加进去、
  重新生成 mesh_offsets.json 再上传——那一步依赖本地 `.motive` 文件,别在 HPC 上折腾
- person 命名不统一(有的 take 是 p1/p2),按实际文件名改
- ufbx 是纯 pip 包,无系统依赖;rtree 需要 libspatialindex,pip 的 wheel 通常自带,
  装不上就 `conda install rtree` 或找管理员
