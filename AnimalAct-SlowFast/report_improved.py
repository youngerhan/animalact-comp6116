"""Recompute the final fixed-model comparison from saved scores and validation thresholds."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from clips import read_json,write_json
from metrics import evaluate,confusion_counts,f1_scores
from runtime import PROJECT

OUT=PROJECT/'results/improvement-20261125'


def counts(report):
    result={k:sum(r[k] for r in report['per_class']) for k in ('TP','FP','FN','TN')}
    tp,fp,fn=result['TP'],result['FP'],result['FN']
    result.update(mAP=report['mAP'],macro_F1=report['macro_F1'],precision=tp/(tp+fp),recall=tp/(tp+fn),micro_F1=2*tp/(2*tp+fp+fn))
    return result


def main():
    manifest=read_json(PROJECT/'data/main.json');classes=manifest['classes']
    run=PROJECT/'runs/improved-swin-final-20261125';oldrun=PROJECT/'runs/pilot32-blend'
    threshold=np.asarray(read_json(OUT/'calibration-final/selection-frozen.json')['thresholds'])
    for split in ('valid','test'):
        rows=manifest['splits'][split]
        with np.load(run/f'{split}_predictions.npz') as z:
            targets=np.zeros_like(z['targets'])
            for i,row in enumerate(rows):targets[i,row['labels']]=True
            assert np.array_equal(z['targets'],targets)
            for key,values in [('global_class_ids',[c['global_id'] for c in classes]),('video_ids',[r['video_id'] for r in rows]),('start_frames',[r['start_frame'] for r in rows]),('end_frames',[r['end_frame'] for r in rows])]:
                assert np.array_equal(z[key],values),(split,key)
    with np.load(run/'test_predictions.npz') as z:y,s=z['targets'],z['scores']
    with np.load(oldrun/'test_predictions.npz') as z:
        assert np.array_equal(y,z['targets']);before_scores=z['scores']
    oldreport=read_json(oldrun/'test_metrics.json');uniform=read_json(run/'test_metrics.json')
    old=evaluate(y,before_scores,classes,oldreport['threshold'])
    new=evaluate(y,s,classes,threshold)
    assert abs(new['mAP']-uniform['mAP'])<1e-12
    calibrated=read_json(OUT/'calibration-final/test-calibrated-metrics.json')
    assert abs(new['macro_F1']-calibrated['macro_F1'])<1e-12
    rows=manifest['splits']['test'];ids=np.array([r['video_id'] for r in rows]);videos=np.unique(ids)
    groups=[np.flatnonzero(ids==v) for v in videos];rng=np.random.default_rng(42);deltas=[]
    for _ in range(300):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        oldf=f1_scores(*confusion_counts(y[ix],before_scores[ix],oldreport['threshold'])[:3]).mean()
        newf=f1_scores(*confusion_counts(y[ix],s[ix],threshold)[:3]).mean()
        deltas.append(100*(newf-oldf))
    ci=read_json(OUT/'summary-swin.json')['paired_video_bootstrap_95_interval_pp']['mAP']
    summary=dict(evaluation='31-class annotated-interval multilabel classification',test_clips=len(y),test_videos=len(videos),before=counts(old),after_uniform=counts(uniform),after_calibrated=counts(new),delta_pp={k:100*(counts(new)[k]-counts(old)[k]) for k in ('mAP','macro_F1','precision','recall','micro_F1')},paired_video_bootstrap_95_interval_pp=dict(mAP=ci,calibrated_macro_F1=np.percentile(deltas,[2.5,97.5]).tolist()),bootstrap_repeats=300,validation_only_selection=True,exploratory_test_exposure='Two planned stages and later validation-selected threshold calibration evaluated test; no single-use untouched-test claim.',new_model_count=10,old_model_count=8)
    write_json(OUT/'summary-final.json',summary)
    with (OUT/'per-class-final.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['name','positives','AP_before','AP_after','AP_delta_pp','F1_before','F1_after','F1_delta_pp','TP','FP','FN','TN','threshold']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,(a,b) in enumerate(zip(old['per_class'],new['per_class'])):
            writer.writerow(dict(name=b['name'],positives=b['positives'],AP_before=a['AP'],AP_after=b['AP'],AP_delta_pp=100*(b['AP']-a['AP']),F1_before=a['F1'],F1_after=b['F1'],F1_delta_pp=100*(b['F1']-a['F1']),**{k:b[k] for k in ('TP','FP','FN','TN')},threshold=threshold[i]))
    a,b=summary['before'],summary['after_calibrated']
    lines=['# AnimalAct 本轮训练最终结果（2026-11-25）','',
    '本轮完成三组 SlowFast 对照、一个 Video Swin-T 模型及验证集阈值校准。mAP 小幅提高，Macro-F1 基本持平，不能称为大幅提升。原始标注、类别和数据划分没有更改。','',
    '## 同一测试集的最终对照','',
    '| 指标 | 原融合 | 新融合＋阈值校准 | 变化（百分点） |','|---|---:|---:|---:|']
    for key,label in [('mAP','mAP'),('macro_F1','Macro-F1'),('precision','微平均 Precision'),('recall','微平均 Recall'),('micro_F1','Micro-F1')]:
        lines.append(f'| {label} | {a[key]:.4%} | {b[key]:.4%} | {100*(b[key]-a[key]):+.4f} |')
    lines+=['',f"误报 FP：{a['FP']} → {b['FP']}；漏报 FN：{a['FN']} → {b['FN']}；正确检出 TP：{a['TP']} → {b['TP']}。",
    '',f"新融合统一阈值 0.15 的 Macro-F1 为 {uniform['macro_F1']:.4%}；正则化逐类阈值使其达到 {new['macro_F1']:.4%}。阈值不改变连续分数，因此不改变 mAP。原融合使用统一阈值，新方案的校准必须在汇报中注明；公平对照应同时展示未校准结果。",
    '', '## 训练与选定方案','',
    '| 尝试 | 实际训练 | 结果及处理 |','|---|---|---|',
    '| 32 帧训练前后文增强 | 4 轮，最佳第 2 轮 | 保留；只在训练扩展到至少 0.75 秒，评估仍限于原片段 |',
    '| 运动区域输入 | 4 轮 | 验证表现较弱，未进入最终融合；运动差分不是动物检测器 |',
    '| 非对称损失 ASL | 2 轮后早停 | 最佳仍为起始权重，未进入最终融合 |',
    '| Kinetics-400 预训练 Video Swin-T | 冻结编码器拟合头 33 轮＋末两阶段微调 6 轮 | 最佳微调第 6 轮；以 12.5% 加入融合 |',
    '| 逐类融合权重 | 验证集按视频分组交叉验证 | 未通过采用条件，保留统一融合权重 |',
    '| 逐类阈值 | 验证集 5 折，正则强度 300 | 保留；少样本阈值向全局阈值收缩 |','',
    '最终配方：0.875 ×（0.75 × 原融合＋0.25 × 前后文模型翻转平均）＋0.125 × Swin 翻转平均。展开后为 10 个模型：7 个旧成员各 0.046875，旧 32 帧模型 0.328125，前后文模型 0.21875，Swin 0.125。推理成本随模型数量和翻转平均增加。',
    '', '## 指标可信度与适用范围','',
    f"测试集共 {len(y)} 个已标注区间片段、31 类、覆盖 {len(videos)} 个视频。它衡量区间分类，尚不包含未知长视频的自动动作定位；也不是全部 447 个官方测试视频的整视频评测。",
    '',f"按视频成组配对重采样 300 次，mAP 改变量 95% 分位区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] 个百分点；校准后 Macro-F1 改变量区间 [{summary['paired_video_bootstrap_95_interval_pp']['calibrated_macro_F1'][0]:+.2f}, {summary['paired_video_bootstrap_95_interval_pp']['calibrated_macro_F1'][1]:+.2f}]。两者均跨过 0，没有足够证据称为稳定或显著提升。这也不衡量重新训练的随机种子波动。",
    '', '检查点、翻转、融合和阈值均由验证集选择。本轮分阶段查看过测试结果，且此前已有多轮历史实验；这是探索性比较，不能宣称测试集从未被查看。阈值交叉验证条件于已选模型，不是对整个模型选择过程的嵌套验证。部分历史验证预测缺少身份元数据：标签能核对，但不能完全证明同标签区间的顺序。新融合完整元数据已与清单核对。',
    '', '## 推理复现与文件','',
    '旧 ft256 模型的历史中心裁剪使用向下取整；后续代码使用四舍五入，在部分尺寸相差 1 像素。新增 runs/ft256/inference_overrides.json 单独保留历史方式，未改变其他模型和训练默认行为。整套核验结果见 legacy-inference-verification.json，实际十模型单片段核验见 inference-verification.json。',
    '', 'FINAL_REPORT.md 是本轮总报告；REPORT.md 是第一阶段记录。summary-final.json 保存最终原始数值，per-class-final.csv 保存逐类 AP、F1 和混淆计数，calibration-final/selection-frozen.json 保存阈值。训练日志及预先写定的计划在本目录；源码和检查点在项目根目录及 runs/。',
    '', '原复现 ZIP 与 GitHub v1.0.0 发布包对应本轮训练之前的版本；本轮通过 v1.1.0 增量包提供，先安装旧完整包再应用增量。历史版本保持不变。','']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
