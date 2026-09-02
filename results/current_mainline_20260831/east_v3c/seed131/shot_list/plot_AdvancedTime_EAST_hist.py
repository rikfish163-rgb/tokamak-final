import json
from pathlib import Path

import matplotlib.pyplot as plt

# JSON 文件路径（相对当前脚本所在目录）
json_path = Path('EAST') / 'AdvancedTime_EAST.json'

with json_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

# 取出所有时间值
values = list(data.values())

plt.figure(figsize=(8, 5))
plt.hist(values, bins=30, edgecolor='black', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('AdvancedTime_EAST Distribution')
plt.grid(axis='y', alpha=0.3)

out_path = Path('EAST') / 'AdvancedTime_EAST_hist.png'
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f'Histogram saved to: {out_path.resolve()}')
