"""Attach visual review notes and a concise audit report to generated evidence."""
import csv
import hashlib
import html
import json
from collections import Counter,defaultdict
from pathlib import Path

import argparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    project,out=args.project,args.out
    cases=json.loads((out/'cases.json').read_text(encoding='utf-8'))
    notes={
    1:('细节与动作相似','猛禽与其他鸟同框，嘴部细节较小；模型输出 Preening/Sensing。仅凭抽样帧不能确认叼物判定，需要看片复核。'),
    2:('远景与疑似外观依赖','坡地上的动物嘴部很小，模型却高分输出 Carrying In Mouth；标注仅 Sensing。是否使用了外观捷径需要裁剪对照确认。'),
    3:('遮挡与暗光','动物交互发生在枝叶和阴影中，模型输出 Keeping still，难以看清被捕食/咬合细节。'),
    4:('短时细微动作','区间只有 14 个不同原始帧；Eating 被检出而 Shaking 漏检。增加重复采样帧不能增加真实运动信息。'),
    5:('极短区间与近义动作','区间仅 3 个原始帧，模型输出 Shaking/Shaking Head 等多个动作；GT 包含 Preening 等。无法由 32 帧输入恢复更长时间运动。'),
    6:('行为迁移差异','红色动物在毛发表面移动，标注 Climbing，模型输出 Eating。主体可见，不能简单归因于动物太小。'),
    7:('远景与背景线索候选','多只猴在树林地面，模型 Climbing 0.977。树木背景与动作可能混淆，但未做背景遮挡实验，不能确认因果。'),
    8:('夜间噪声与细节','红外夜景、地面动物与枝叶纹理混杂；Grooming 被预测为 Eating。'),
    9:('极短区间','Running 区间只有 3 帧，模型预测 Grooming。须结合周围时间上下文检查，不能仅调阈值解决。'),
    10:('交互动作与单体运动','追逐场景中 Running 被检出，Attacking 漏检；交互关系比单体移动更难表达。'),
    11:('相似交互行为','虎狮交互，原标注 Disturbing Another Animal，模型 Attacking 高分。属于应人工复核的近义边界，不能直接宣布标注错。'),
    12:('细粒度语义混淆','蛇抬头画面，Attending 漏检，输出 Sensing/Keeping still。抽样帧无法清楚区分这些概念。'),
    13:('多标签覆盖候选','猫头鹰画面，Chirping 已正确检出，模型额外输出 Attending/Keeping still；是否应共存需按课程定义复核。'),
    14:('动作泛化','鬣蜥近景、主体大，Sensing 被预测为 Walking/Climbing。该例不支持小主体作为唯一解释。'),
    15:('极短区间与语义混淆','植被中的鸟只取 3 个原始帧，标注 Turning Around 等而模型额外给 Sensing。'),
    16:('小主体与并行动作','蚂蚁/小型昆虫与地面纹理混杂；GT 含 Eating/Keeping still/Moving，模型仅强输出 Moving。多动物或并行动作不容易同时保留。'),
    17:('标注覆盖复核候选','鸟停在枝上，8 个抽样帧姿态大致稳定。模型 Keeping still 0.988，但当前片段仅标 Attending，且该视频原标注完全没有 Keeping still。值得人工复核，尚不能判错。'),
    18:('并行动作细节','水中动物 Swimming 被检出、Eating 漏检；嘴部与水面遮挡限制细粒度观察。'),
    19:('短区间与物种外观候选','植被中鸟的 Flapping 被检出，模型额外给 Eating 0.961。区间仅 8 帧，不排除背景/外观依赖。'),
    20:('并行动作与感知通道','鹅摇头 8 帧区间中，Shaking Head 被检出而 Chirping 漏检。当前模型只有 RGB，抽样图与无音频检查无法确认鸣叫。'),
    21:('同场景跨时间混淆','夜莺视频当前标注 Attending/Keeping still，但模型 Chirping 0.981；鸣叫在该视频其他区间有标注。不能据此断定当前区间也有鸣叫。'),
    22:('动作粒度与物种迁移','蝾螈占据较大画面，标注 Walking，模型 Moving 接近 1、Walking 接近 0。这不是主体太小；需检查动作粒度与训练物种分布。'),
    23:('远景细节不足','远处鸟类呈小轮廓，GT 为 Keeping still/Preening，模型预测 Walking/Sensing；腿部和梳羽细节难辨。'),
    24:('目标动作正确对照','火烈鸟饮水场景，目标 Drinking 被高分检出；这是目标类别正确对照。'),
    25:('目标动作正确对照','眼镜蛇 Spitting Venom 目标被高分检出；仅 10 个原始帧。不能用单例证明模型一定依据毒液而非其他线索。'),
    26:('部分正确对照','鸟类 Walking 被检出，但仍有漏标/额外预测。这里只作为 Walking 正确对照，并非整个标签集合完全正确。'),
    }
    for case in cases:
        tag,note=notes[case['case_id']]
        case.update(review_tag=tag,visual_review_note=note,review_method='8 uniformly sampled frames; original-resolution middle frames additionally checked for cases 2,7,17,22; no audio verification')
    (out/'cases-reviewed.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source=project.parent/'AnimalAct'; counts=defaultdict(Counter)
    for sample in json.loads((source/'splits/train.json').read_text(encoding='utf-8'))['samples']:
        annotation=json.loads((source/sample['annotation_path']).read_text(encoding='utf-8'))
        actions={action['action'] for animal in annotation['annotations'] for action in animal['actions']}
        for action in ('Walking','Moving'):
            if action in actions: counts[action][sample['parent_class']]+=1
    (out/'training-action-animal-groups.json').write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
    motion=json.loads((out/'motion-probe.json').read_text(encoding='utf-8'))
    with (out/'motion-per-class.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['action','normal_AP','static_AP','normal_minus_static_pp'])
        for a,b in zip(motion['normal']['per_class'],motion['static_middle_frame']['per_class']):
            w.writerow([a['name'],a['AP'],b['AP'],100*(a['AP']-b['AP'])])
    report='''# AnimalAct 错误诊断（2026-11-25）

    结论：当前问题由模型混淆、视觉信息不足、数据分布和待复核的标注覆盖共同构成。已有证据不支持“全部是数据问题”，也不支持“模型只看静态画面”。本次只做诊断，没有训练或修改正式权重、阈值、标注和测试结果。

    ## 检查方法与范围

    - 以最终融合的验证集预测为主：912 个片段、417 个视频、31 类、阈值 0.15。测试集仅保留原有分数的描述性统计，没有用于选择新模型参数。
    - 定向抽取 26 个不同验证视频：23 个误报/漏报案例和 3 个目标类别正确对照（正确对照不表示所有标签都正确）。每例检查 8 个均匀采样帧；4 例另看原分辨率中间帧。抽样不是随机调查，不能把案例比例当作总体原因比例；没有核听音轨。
    - 复核标签与原始区间的映射。全部已统计 FP/FN 均与原有“覆盖片段至少一半”的规则一致，未发现映射不一致。
    - 额外对固定的 pilot32-whole224 模型做全验证集输入干预：正常 32 帧 vs 将其中第 17 帧重复 32 次。两者权重、224 px 整帧处理、样本顺序及阈值 0.05 完全相同。正常输入分数与原有保存分数逐元素一致。

    ## 量化发现

    | 验证集观察 | 数值 | 能说明什么 |
    |---|---:|---|
    | 最终融合 TP / FP / FN | 903 / 1000 / 520 | 误报数量多，需要同时控制误报和漏报 |
    | Sensing、Keeping still、Attending 的 FP | 505，占全部 FP 的 50.5% | 这三类是优先检查对象；不等于已证明标注造成了这些错误 |
    | FP 与该动作的原标注有部分时间重叠 | 87，占 8.7% | 时间边界/50%规则可能影响部分误报，不能解释大多数误报 |
    | FP 与该动作原标注没有时间重叠 | 913，占 91.3% | 不应仅通过调整区间边界解决 |
    | 验证片段原始帧数不足 32 | 368 / 912，40.4% | 32 帧输入中存在重复采样；增加输入帧数不等于增加真实信息 |
    | 验证片段原始帧数不足 8 | 72 / 912，7.9% | 极短片段尤其缺乏可供判断的运动变化 |

    “原标注未覆盖”不等于“画面绝对不存在”：标注覆盖是否充分需要按课程动作定义人工复核。

    ## 固定权重运动对照（单模型验证集，非最终融合）

    | 输入 | mAP | Macro-F1 |
    |---|---:|---:|
    | 正常 32 帧 | 43.99% | 44.06% |
    | 单帧重复 32 次 | 25.63% | 24.39% |
    | 移除真实帧间变化后的下降 | 18.36 个百分点 | 19.67 个百分点 |

    正常输入 mAP 相对静态输入高约 71.6%。按视频成组配对重采样 300 次，mAP 差的 95% 分位区间为 [14.23, 22.37] 个百分点。这只描述固定模型在这批验证视频上的差异，不代表多次重训的不确定性。重复单帧也是输入分布干预，不能据此量化所有错误的成因。

    运动信息明显有用，当前模型已经学到了部分运动特征；该实验不能证明现有采样或网络已经最优。Flying AP 从 70.60% 降到 19.54%，Landing 从 49.71% 降到 14.33%，支持继续保留视频建模。

    ## 典型案例证据

    | 案例 | 观察 | 判断与限制 |
    |---|---|---|
    | #7 BSGIHFFP | 树林地面的猴，模型 Climbing 0.977；GT 为 Keeping still/Walking | 主体小，树木背景可能干扰；背景依赖尚未通过遮挡对照确认 |
    | #17 BLGTINQN | 枝上鸟姿态大致稳定，模型 Keeping still 0.988；GT 仅 Attending | 标注覆盖复核候选，不能直接判定原标注错误 |
    | #22 YVFBNICC | 近景蝾螈，模型 Moving 接近 1，Walking 接近 0；GT 为 Walking | 主体并不小，提示动作粒度/物种迁移问题 |
    | #5 WHMIKMRN | 区间只有 3 帧，模型输出多种摇动/梳羽行为 | 原始时间信息极少，重复补成 32 帧无法恢复动作过程 |
    | #10 QISNEUDX | 追逐画面中 Running 正确，Attacking 漏检 | 交互关系与并行动作比单体运动更难 |
    | #20 VFHQMNCR | 摇头被检出，鸣叫漏检 | RGB 模型可能缺乏鸣叫线索；本次未核听声音，无法确认具体原因 |

    各案例完整原标注、逐类预测和图像在 cases-reviewed.json 与 index.html。

    ## 训练数据中的动作与动物类别相关性

    按原始训练视频的 parent_class 统计，每个动作在每个视频只计一次（不同动作之间可重叠）：

    | 动作 | 哺乳类 | 鸟类 | 爬行类 | 两栖类 | 昆虫类 | 合计 |
    |---|---:|---:|---:|---:|---:|---:|
    | Walking | 128 | 127 | 35 | 23 | 0 | 313 |
    | Moving | 4 | 3 | 101 | 19 | 145 | 272 |

    Walking 视频的 81.5% 属鸟类或哺乳类；Moving 的 90.4% 属昆虫或爬行类。数据中存在明显相关性，但这还不能证明模型一定依赖物种捷径。#22 是一个值得用动物区域/背景对照进一步验证的案例。

    ## 建议的下一轮实验顺序

    1. **先做小主体与背景对照。** 在训练集确定固定的动物区域输入策略，与整帧分支融合；相同初始化、训练预算和验证集比较整体 mAP、Macro-F1及每类 AP。不能只在手工挑选错误案例上测试，也不能把人工框当作自动部署效果。
    2. **再做短片段上下文对照。** 原片段标签不变，增加片段周围的时间上下文，对照当前区间内均匀采样。分别汇报原始帧不足 8、8–31、至少 32 的组别。上下文可能引入其他动作，因此需同时观察 FP 是否增加。
    3. **复核训练集标签覆盖和困难类别。** 根据课程定义检查 Attending/Sensing/Keeping still，以及 Walking/Moving 等；记录疑点，获得明确依据后才创建独立的训练标签版本。保留官方验证/测试标签，修订口径如要评价必须单列。

    下一轮首先在验证集选择方案；不要根据测试集的某几例改阈值或标签。没有证据支持承诺涨到某个分数。本次正式成绩仍为测试 mAP 48.38%、Macro-F1 46.67%。

    ## 文件与复现

    - index.html：26 例图像、原视频链接、视觉检查备注及原标注。
    - all-errors.csv：验证/测试每个 FP/FN 的分数、区间覆盖和视频编号。
    - summary.json：错误计数、时间重叠及同片段 FN/FP 共现。
    - motion-probe.json / motion-probe.npz / motion-per-class.csv：运动对照原始结果。
    - training-action-animal-groups.json：训练动作/动物主类别计数。
    - audit_errors.py / probe_motion.py：可重复运行的诊断入口；输出目录请使用新路径。

    ```powershell
    $python='D:\\AnimalAct-setup\\envs\\animalact\\python.exe'
    $project='C:\\Users\\ADMIN\\Desktop\\作业与项目\\特选论题1\\项目\\AnimalAct-SlowFast'
    & $python audit_errors.py --project $project --out ./audit-repeat
    & $python probe_motion.py --project $project --out ./audit-repeat
    ```

    视觉备注是本次对已选样本的检查记录，不是由脚本自动产生的真实标签。
    '''
    import inspect
    (out/'REPORT.md').write_text(inspect.cleandoc(report)+'\n',encoding='utf-8')
    page=(out/'index.html').read_text(encoding='utf-8')
    for case in cases:
        token=f'<article><h2>#{case["case_id"]} '
        insertion=f'<article><p><strong>{html.escape(case["review_tag"])}</strong>：{html.escape(case["visual_review_note"])}</p><h2>#{case["case_id"]} '
        page=page.replace(token,insertion)
    page=page.replace('<h1>AnimalAct 典型错误与正确对照</h1>', '<h1>AnimalAct 典型错误与正确对照</h1><p>26 个验证视频定向检查；固定权重运动对照：正常视频 mAP 43.99%，重复单帧 25.63%。正式融合成绩保持不变。</p><p><a href="REPORT.md">完整诊断报告</a></p>')
    (out/'index.html').write_text(page,encoding='utf-8')
    digests={}
    for path in [project/'data/main.json',project/'runs/pilot32-blend/valid_predictions.npz',project/'runs/pilot32-whole224/best.pth']:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        digests[str(path)]=digest.hexdigest()
    (out/'input-sha256.json').write_text(json.dumps(digests,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Attached 26 visual reviews, report, species counts and input hashes.')


if __name__ == "__main__":
    main()
