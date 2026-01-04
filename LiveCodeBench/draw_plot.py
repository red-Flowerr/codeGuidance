import os
import json
import re
import matplotlib.pyplot as plt

# 目录路径
base_dir = "/mnt/bn/tiktok-mm-5/aiic/users/tianshun/Evaluation/LiveCodeBench/output"

# 筛选条件
valid_exps = {"v2_old_text"}
valid_steps = [306,612,918,1224,1530]

# 存储数据
data_dict = {exp: {"steps": [], "pass@1": [], "pass@5": [], "pass@10": []} for exp in valid_exps}

# 遍历目录
for folder in sorted(os.listdir(base_dir)):
    match = re.match(r"(v2_old_text)_80k_(\d+)", folder)
    if match:
        exp, step = match.groups()
        step = int(step)
        if exp in valid_exps and step in valid_steps:
            eval_file = os.path.join(base_dir, folder, "Scenario.codegeneration_10_0.2_eval.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        entry = data[0]
                        data_dict[exp]["steps"].append(step)
                        data_dict[exp]["pass@1"].append(entry.get("pass@1", 0) * 100)  # 转为百分比
                        data_dict[exp]["pass@5"].append(entry.get("pass@5", 0) * 100)
                        data_dict[exp]["pass@10"].append(entry.get("pass@10", 0) * 100)

# 对每组实验按 step 排序
for exp in valid_exps:
    steps = data_dict[exp]["steps"]
    sorted_indices = sorted(range(len(steps)), key=lambda i: steps[i])
    for metric in ["steps", "pass@1", "pass@5", "pass@10"]:
        data_dict[exp][metric] = [data_dict[exp][metric][i] for i in sorted_indices]

# 绘图函数
def plot_pass(metric):
    plt.figure(figsize=(8,5))
    for exp in valid_exps:
        steps = data_dict[exp]["steps"]
        values = data_dict[exp][metric]
        plt.plot(steps, values, linestyle='--', label=exp)
    plt.xlabel("Step (epoch)")
    plt.ylabel(f"{metric} (%)")  # y轴带单位
    plt.title(f"Livecodebenchv6 {metric} (CodeGeneration)")
    plt.xticks(valid_steps)
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_png = os.path.join(base_dir, f"{metric}_comparison.png")
    plt.savefig(output_png)
    plt.close()
    print(f"Saved {metric} comparison plot to {output_png}")

# 分别绘制 pass@1, pass@5, pass@10
for metric in ["pass@1", "pass@5", "pass@10"]:
    plot_pass(metric)
