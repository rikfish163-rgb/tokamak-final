"""Verify full four-arm artifacts and write a bounded development result table."""
import argparse
import hashlib
import json
from pathlib import Path


def report(directory):
    directory = Path(directory)
    arms = json.loads((directory/'summary.json').read_text())
    order = json.loads((directory/'order.json').read_text())
    expected = [r['shot'] for r in order]
    if len(expected) != 360 or len(set(expected)) != 360:
        raise ValueError('not a full unique 360-shot manifest')
    lines = ['# Simple23：完整EAST360开发回放', '',
             '四臂同源模型、同排序、同阈值；及时检出要求lead严格>30ms。', '',
             '| 方法 | 及时TP/180 | FP/180 | TPR | FPR | J | 炮间接受次数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('static', 'intra', 'inter', 'both'):
        arm = arms[name]
        if [r['shot'] for r in arm['rows']] != expected:
            raise ValueError(f'{name}: order mismatch')
        m = arm['metrics_full']
        if m['n_disrupt'] != 180 or m['n_safe'] != 180:
            raise ValueError(f'{name}: wrong full-stream denominators')
        tp = sum(r['timely'] for r in arm['rows'])
        fp = sum(r['alarm'] and not r['is_disrupt'] for r in arm['rows'])
        if tp != m['tp'] or fp != m['fp']:
            raise ValueError('counts disagree with per-shot records')
        for index, sid in enumerate(expected, 1):
            receipt = json.loads((directory/name/f'{index:04d}_{sid}_seal.json').read_text())
            digest = hashlib.sha256((directory/name/receipt['npz']).read_bytes()).hexdigest()
            if digest != receipt['sha256']:
                raise ValueError('sealed prediction hash mismatch')
        accepted = sum(r['persistent']['accepted'] for r in arm['rows'])
        lines.append(f"| {name} | {tp} | {fp} | {m['tpr']:.6f} | {m['fpr']:.6f} | {m['J']:.6f} | {accepted} |")
    lines += ['', '## 配对翻转', '', '| 比较 | 救回 | 改坏 | 净救回 |', '|---|---:|---:|---:|']
    for before, after in [('static', 'intra'), ('static', 'inter'), ('inter', 'both')]:
        helpful = harmful = 0
        for a, b in zip(arms[before]['rows'], arms[after]['rows']):
            if a['is_disrupt'] != b['is_disrupt']:
                raise ValueError('outcome differs between arms')
            ca = a['timely'] if a['is_disrupt'] else not a['alarm']
            cb = b['timely'] if b['is_disrupt'] else not b['alarm']
            helpful += not ca and cb
            harmful += ca and not cb
        delta = arms[after]['metrics_full']['J'] - arms[before]['metrics_full']['J']
        if abs(delta - (helpful-harmful)/180) > 1e-10:
            raise ValueError('paired flips disagree with J change')
        lines.append(f'| {before} → {after} | {helpful} | {harmful} | {helpful-harmful} |')
    lines += ['', '## 解释边界', '',
              '- 这是零EAST标签起步的开发回放，不是独立未来集；排序使用起始参数代理，不是实际损伤等级。',
              '- 每炮结局自然揭示；没有标签购买预算。前20炮和后340炮采用不同学习率。',
              '- 配对翻转是相同炮的结果对照；inter/both长期持久状态可能分叉，不能单凭差值断言每个翻转的直接机制。',
              '- 固定步长基线尚未实现自主选择更新窗口与力度；炮内仅损失下降不能证明预警收益。',
              '- 全部1440份封存预测文件的SHA与收据已核对；检查hash不等于独立验证所有因果语义。', '']
    path = directory/'RESULTS.md'
    path.write_text('\n'.join(lines))
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    print(report(args.directory))
