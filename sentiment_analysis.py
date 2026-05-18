"""
革命日记情感分析管道 v2.0 — 增强版
====================================
方法：三级强度领域词典 + 否定词处理 + 程度副词调整 + 情感子类分析
输出：parsed_data/sentiment_results.json

使用：
  python3 sentiment_analysis.py

向后兼容说明：
  - 所有原有字段（score/label/pos_words/neg_words/pos_count/neg_count/snownlp）保持不变
  - 新增字段：intensity / emotion_profile / negations 等
  - UI 无需修改即可正常工作
"""

import os, json, re, pickle, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "parsed_data/sentiment_results.json")

# ============================================================
# 1. 三级强度领域情感词典（革命日记语境）
#    weight: strong=2.0, medium=1.0, weak=0.5
# ============================================================

STRONG_POSITIVE = [
    # 重大军事胜利
    "大捷", "全歼", "聚歼", "凯旋", "光复", "克复",
    "胜利会师", "告捷",
    # 崇高评价
    "伟大胜利", "决定性胜利", "辉煌", "丰功伟绩",
    "永垂不朽", "万古流芳",
]

MEDIUM_POSITIVE = [
    # 军事行动 — 胜利/攻克
    "胜利", "攻克", "收复", "解放", "占领", "攻占", "夺取", "突破",
    "击溃", "击退", "击毙", "俘虏", "缴获", "歼灭", "围歼",
    "突围", "反攻", "反击", "出击",
    "俘敌", "歼敌", "消灭敌人",
    # 积极评价
    "光荣", "荣誉", "模范", "英雄", "功臣", "先进", "优秀", "出色",
    "卓越", "伟大", "崇高", "英勇", "英明", "正确",
    "表率", "先锋", "骨干", "榜样", "好样",
    # 正面情感
    "幸福", "高兴", "快乐", "喜悦", "欢喜", "兴奋", "激动", "感动",
    "满意", "欣慰", "鼓舞", "振奋", "自豪", "骄傲", "光荣",
    "满意", "愉快",
    # 正面行动
    "进步", "发展", "提高", "增长", "扩大", "推广", "繁荣",
    "富强", "兴旺", "昌盛", "复兴",
    # 团结与支持
    "团结", "拥护", "支持", "赞成", "同意", "响应", "配合", "协作",
    "合作", "互助", "友爱", "关怀", "关心", "爱护",
    "拥军", "爱民", "优属", "拥政爱民", "军民一家",
    "统一战线",
    # 称赞与奖励
    "称赞", "表扬", "表彰", "奖励", "嘉奖", "嘉勉", "祝贺", "庆祝",
    "致敬", "慰问", "感谢", "感激", "感恩",
    # 希望与信心
    "希望", "信心", "信念", "决心", "斗志", "勇气", "干劲", "热情",
    "积极", "乐观", "光明", "前途",
    # 和平与安宁
    "和平", "安宁", "安定", "稳定", "安全", "和谐", "太平", "安康",
    # 建设成就
    "丰收", "增产", "自给", "自足", "丰衣足食", "丰衣",
    "好转", "改善", "解决",
    "开荒", "生产", "发展生产", "大生产",
    "组织起来", "互助合作", "生产节约",
    # 革命精神
    "斗志昂扬", "士气高涨", "同仇敌忾", "万众一心", "众志成城",
    "请战", "求战", "誓师",
    # 政策正面
    "减租减息", "土地改革", "民主", "自治",
    # 正面自然/生活
    "很好", "美好", "良好", "优良", "不少",
]

WEAK_POSITIVE = [
    "还好", "不错", "尚好", "顺利", "通畅", "如意",
    "轻松", "舒适", "温暖", "充足", "丰富", "富裕",
    "安心", "放心", "宽心",
    "好的", "好了",
    "更好", "更好", "较好",
    "帮助", "协助", "帮忙",
]

STRONG_NEGATIVE = [
    # 重大失败
    "惨败", "溃败", "覆灭", "覆没", "全军覆没",
    "沦陷", "失守", "丢失", "失陷",
    "崩溃", "瓦解", "溃散", "溃退",
    # 敌人暴行
    "屠杀", "杀戮", "残杀", "杀害", "毒杀",
    "烧杀", "奸淫", "掳掠",
    "三光", "抢光", "杀光", "烧光",
    "洗劫", "劫掠", "掠夺",
    # 严重灾难
    "毁灭", "摧毁", "灭绝",
    # 严重损失
    "牺牲", "阵亡", "殉国", "殉职", "捐躯",
    "伤亡惨重",
    # 极度负向评价
    "残暴", "残忍", "凶残", "恶毒", "阴险", "狠毒", "毒辣",
]

MEDIUM_NEGATIVE = [
    # 困难与艰苦
    "困难", "艰苦", "艰辛", "艰难", "困苦", "苦难", "艰巨",
    "曲折", "坎坷", "逆境",
    "艰苦卓绝", "千辛万苦", "跋山涉水",
    # 负面情感
    "痛苦", "悲伤", "悲哀", "悲痛", "忧伤", "忧愁", "忧虑",
    "焦虑", "担心", "担忧",
    "恐惧", "害怕", "惊慌", "恐慌", "惊恐", "畏惧", "惧怕",
    "紧张", "不安", "无奈", "绝望",
    "愤怒", "气愤", "愤慨", "憎恨", "仇恨", "讨厌", "厌恶", "反感",
    "后悔", "惋惜", "遗憾", "失望",
    "苦闷", "苦恼", "烦恼", "烦躁", "沉重",
    "心焦", "心乱",
    # 牺牲与损失（不重复 strong 中已有的词）
    "伤亡", "损耗", "消耗", "浪费",
    "失败", "失利", "受挫", "挫折", "打击",
    "缺点", "问题",
    # 伤病
    "疾病", "病倒", "病重", "生病", "病死",
    "受伤", "负伤", "流血",
    # 敌人与威胁（不重复 strong 中的词）
    "敌人", "反动", "侵略", "侵犯", "袭击", "偷袭",
    "围攻", "围剿", "扫荡", "清乡", "蚕食",
    "镇压", "压迫", "剥削", "欺压", "迫害", "摧残",
    "封锁", "围困",
    "特务", "汉奸", "伪军", "鬼子",
    "顽固", "顽固派",
    # 短缺与贫乏
    "缺少", "缺乏", "不足", "短缺", "贫乏", "贫穷", "贫困",
    "穷困", "赤贫",
    "荒凉", "荒芜", "贫瘠", "落后",
    "缺粮", "缺盐", "缺药", "缺衣", "缺钱",
    "断粮", "断炊", "饿饭", "饥荒", "饥饿",
    # 危机与动荡
    "危机", "危险", "风险", "困境", "绝境",
    "动荡", "混乱", "冲突", "摩擦",
    # 灾难
    "灾难", "灾害", "祸害", "不幸", "悲惨", "惨痛",
    "逃难", "逃亡", "流浪", "流亡",
    # 消极状态
    "消极", "悲观", "低落", "消沉", "颓废",
    "懒散", "松懈", "涣散",
    "腐败", "贪污", "浪费", "奢侈",
    "官僚主义", "形式主义", "教条主义",
    # 恶劣环境（军事相关）
    "严寒", "酷暑", "暴雨", "洪水", "大雪",
    "冻伤", "中暑",
    # 负面军事行动
    "叛变", "投敌", "逃跑", "溃逃", "退缩", "怕死",
    "撤退", "退却", "败退",
    # 战斗损失描述
    "死伤", "战死", "打死",
    "死了",
    "惨重",
    # 负面评价
    "不好", "不行", "糟糕", "恶劣",
]

WEAK_NEGATIVE = [
    "疲劳", "疲惫", "疲乏", "劳累", "困乏",
    "干渴", "口渴",
    "困倦", "瞌睡", "失眠",
    "不便", "麻烦",
    "寒冷", "阴冷", "潮湿", "泥泞",
    "单调", "枯燥", "无聊",
    "孤单", "孤独", "寂寞",
    "缓慢", "迟缓", "拖延",
    "勉强", "凑合",
    "忙乱",
    "挂念", "牵挂",
    "惭愧", "内疚", "歉意",
    "可惜", "心疼", "舍不得",
    "睡不好", "睡不安",
    "阴天", "下雨", "风雪",
]

# 合并为带权重的查找词典
def _build_weight_map():
    # 先处理强层级，再处理弱层级（已有键不覆盖，保证强权重优先）
    wmap = {}
    for w in STRONG_POSITIVE:
        if w not in wmap: wmap[w] = 2.0
    for w in STRONG_NEGATIVE:
        if w not in wmap: wmap[w] = -2.0
    for w in MEDIUM_POSITIVE:
        if w not in wmap: wmap[w] = 1.0
    for w in MEDIUM_NEGATIVE:
        if w not in wmap: wmap[w] = -1.0
    for w in WEAK_POSITIVE:
        if w not in wmap: wmap[w] = 0.5
    for w in WEAK_NEGATIVE:
        if w not in wmap: wmap[w] = -0.5
    return wmap

WORD_WEIGHT_MAP = _build_weight_map()

# 用于 UI 展示的平铺词表（保持后向兼容）
POSITIVE_WORDS = STRONG_POSITIVE + MEDIUM_POSITIVE + WEAK_POSITIVE
NEGATIVE_WORDS = STRONG_NEGATIVE + MEDIUM_NEGATIVE + WEAK_NEGATIVE
POSITIVE_SET = set(POSITIVE_WORDS)
NEGATIVE_SET = set(NEGATIVE_WORDS)

# ============================================================
# 2. 情感子类词典（用于情感画像）
# ============================================================
EMOTION_DICT = {
    "喜悦": [
        "高兴", "快乐", "喜悦", "欢喜", "兴奋", "激动", "欢欣", "欢快",
        "欢腾", "欢呼", "欢庆", "欢天喜地", "兴高采烈",
        "幸福", "甜蜜", "舒心", "畅快", "惬意",
        "愉快", "开心",
    ],
    "悲伤": [
        "悲伤", "悲哀", "悲痛", "忧伤", "忧愁", "哀伤", "哀痛",
        "伤心", "难过", "难受", "心酸", "辛酸",
        "痛哭", "哭泣", "流泪", "泣不成声",
        "凄凉", "悲凉", "凄惨",
        "惋惜", "遗憾", "痛心", "沉痛", "沉重",
        "悲痛", "哀悼",
    ],
    "愤怒": [
        "愤怒", "气愤", "愤慨", "愤恨", "憎恨", "仇恨", "痛恨", "憎恶",
        "恼火", "恼怒", "生气", "怒斥", "怒骂",
        "义愤填膺", "满腔怒火",
        "反感", "厌恶", "讨厌",
    ],
    "恐惧": [
        "恐惧", "害怕", "惊慌", "恐慌", "惊恐", "畏惧", "惧怕",
        "紧张", "不安", "心惊胆战", "提心吊胆",
        "担心", "担忧", "忧虑", "焦虑",
        "绝望", "无奈", "无助",
    ],
    "期盼": [
        "希望", "期望", "期待", "盼望", "渴望", "祈望", "指望",
        "决心", "信念", "信心", "斗志", "勇气", "干劲",
        "誓言", "宣誓", "立志", "奋发",
        "前途", "未来", "远景",
    ],
    "信任": [
        "信任", "信赖", "相信", "坚信", "确信", "深信",
        "忠诚", "忠实", "赤诚",
        "拥护", "支持", "赞成", "同意", "响应",
        "可靠", "可信", "稳妥",
    ],
}

# ============================================================
# 3. 否定词和程度副词
# ============================================================
NEGATION_WORDS = {"不", "没", "未", "别", "莫", "休", "无", "非", "勿", "毋"}

INTENSIFIERS = {
    "非常": 1.8, "十分": 1.8, "极其": 2.0, "极为": 1.9, "异常": 1.7,
    "特别": 1.6, "相当": 1.5, "比较": 0.7, "颇": 1.3, "较": 0.8,
    "有些": 0.5, "有点": 0.5, "一点儿": 0.4,
    "更加": 1.6, "更": 1.4, "越": 1.3,
    "太": 1.5, "极": 1.8, "最": 1.7,
    "格外": 1.6, "极其": 2.0,
    "稍微": 0.5, "略微": 0.4,
}

# ============================================================
# jieba 分词（复用领域词典）
# ============================================================
def get_jieba():
    import jieba
    domain_words = [
        "长征", "抗日", "解放战争", "八路军", "新四军", "红军", "革命",
        "南泥湾", "大生产", "陕甘宁", "晋察冀", "根据地",
        "宿营", "行军", "作战", "战斗", "突围", "歼灭", "俘虏",
        "指战员", "同志", "政委", "司令员", "旅长", "团长",
        "群众", "老百姓", "民夫", "老乡", "工作队",
        "土改", "整风", "大生产", "生产", "开荒", "种地",
        "机枪", "步枪", "手榴弹", "炮弹", "子弹",
        "合作社", "公粮", "救国公粮", "慰问", "劳军",
        "冬衣", "粮食", "伙食", "伤病员", "卫生队",
        "党支部", "党员", "支部会议", "组织生活",
        "俘虏", "伪军", "日军", "鬼子", "汉奸",
        "游击战", "运动战", "阵地战", "攻坚战",
        "减租减息", "土地改革", "统一战线",
        "整编", "整训", "整军", "整党",
        "攻克", "收复", "解放", "占领",
        "围歼", "聚歼", "全歼",
        "拥政爱民", "拥军优属",
        "参军", "支前", "劳军",
        "民主", "自治", "政权建设",
        "文化教育", "识字", "扫盲",
        "大生产运动", "生产节约",
    ]
    for w in domain_words:
        jieba.add_word(w)
    return jieba


def tokenize(text, jieba):
    """分词，返回词列表"""
    text = re.sub(r'[\s\n\r]+', ' ', text)
    words = jieba.lcut(text)
    words = [w.strip() for w in words
             if len(w.strip()) > 1
             and not re.match(r'^[，。、；：！？""''（）【】《》' + r'\s\d]+$', w)]
    return words


# ============================================================
# 4. 增强情感评分
# ============================================================
def score_entry(text, jieba):
    """对单条日记进行增强情感评分，返回详细结果"""
    words = tokenize(text, jieba)

    pos_total = 0.0
    neg_total = 0.0
    pos_words_found = []
    neg_words_found = []
    negations_detected = []
    emotion_counts = {cat: 0 for cat in EMOTION_DICT}

    # 构建词到情感子类映射
    word_to_emotion = {}
    for cat, cat_words in EMOTION_DICT.items():
        for w in cat_words:
            word_to_emotion.setdefault(w, []).append(cat)

    for i, word in enumerate(words):
        # --- 情感词典匹配 ---
        if word not in WORD_WEIGHT_MAP:
            continue

        base_weight = WORD_WEIGHT_MAP[word]

        # --- 检查前面2词范围内是否有否定词 ---
        negated = False
        neg_word = None
        for j in range(max(0, i - 2), i):
            if words[j] in NEGATION_WORDS:
                negated = True
                neg_word = words[j]
                break

        # --- 检查前一词是否为程度副词 ---
        degree = 1.0
        if i > 0 and words[i - 1] in INTENSIFIERS:
            degree = INTENSIFIERS[words[i - 1]]

        # --- 计算有效权重 ---
        if negated:
            # 否定翻转：正向变负向，负向变正向，幅度减弱
            effective = -base_weight * 0.6 * degree
            negations_detected.append(word)
        else:
            effective = base_weight * degree

        display_word = f"{neg_word}{word}" if negated else word

        if effective > 0:
            pos_total += effective
            if display_word not in pos_words_found:
                pos_words_found.append(display_word)
        else:
            neg_total += abs(effective)
            if display_word not in neg_words_found:
                neg_words_found.append(display_word)

        # --- 情感子类计数 ---
        if word in word_to_emotion:
            for cat in word_to_emotion[word]:
                emotion_counts[cat] += 1

    # --- 计算得分（保持 [-1, 1] 范围）---
    score = (pos_total - neg_total) / (pos_total + neg_total + 1)

    # --- 标签 ---
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"

    # --- 强度（绝对值放大，0~1）---
    intensity = abs(score)
    # 增强区分度：如果有很多情感词，强度应该更大
    total_sentiment_words = pos_total + neg_total
    if total_sentiment_words > 5:
        intensity = min(1.0, intensity * 1.2)

    result = {
        "pos_count": len(pos_words_found),
        "neg_count": len(neg_words_found),
        "pos_words": pos_words_found,
        "neg_words": neg_words_found,
        "score": round(score, 4),
        "label": label,
        "intensity": round(intensity, 4),
        "pos_intensity": round(pos_total / (pos_total + neg_total + 1), 4),
        "neg_intensity": round(neg_total / (pos_total + neg_total + 1), 4),
        "emotion_profile": emotion_counts,
        "negations": negations_detected,
    }

    # SnowNLP 对照（可选）
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        result["snownlp"] = round(s.sentiments, 4)
    except Exception:
        result["snownlp"] = None

    return result


# ============================================================
# 5. 增强聚合统计
# ============================================================
def aggregate(entries, scores, emotion_profiles=None):
    """按年份、分类、日记本聚合情感得分及强度"""
    by_year = {}
    by_category = {}
    by_diary = {}
    intensity_by_year = {}
    emotion_by_year = {}

    for i, e in enumerate(entries):
        yr = e.get('year')
        cat = e.get('category', '未分类') or '未分类'
        diary = e.get('diary_name', '未知')
        sc = scores[i]['score']

        if yr is not None:
            by_year.setdefault(yr, []).append(sc)
            intensity_by_year.setdefault(yr, []).append(scores[i].get('intensity', abs(sc)))
            if emotion_profiles:
                emotion_by_year.setdefault(yr, {})
                for cat_name, count in emotion_profiles[i].items():
                    emotion_by_year[yr].setdefault(cat_name, []).append(count)

        by_category.setdefault(cat, []).append(sc)
        by_diary.setdefault(diary, []).append(sc)

    def summarize(arr):
        arr = np.array(arr)
        pos_pct = np.mean(arr > 0.05) * 100
        neg_pct = np.mean(arr < -0.05) * 100
        neutral_pct = np.mean((arr >= -0.05) & (arr <= 0.05)) * 100
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "count": len(arr),
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
            "neutral_pct": round(neutral_pct, 1),
        }

    def summarize_intensity(arr):
        arr = np.array(arr)
        return {
            "mean_intensity": round(float(np.mean(arr)), 4),
            "max_intensity": round(float(np.max(arr)), 4),
            "std_intensity": round(float(np.std(arr)), 4),
        }

    # 情感强度聚合
    int_by_year = {}
    for yr, vals in intensity_by_year.items():
        int_by_year[str(yr)] = summarize_intensity(vals)

    # 情感子类聚合
    emo_by_year = {}
    if emotion_profiles:
        for yr, cat_dict in emotion_by_year.items():
            emo_by_year[str(yr)] = {
                cat: round(float(np.mean(vals)), 4)
                for cat, vals in cat_dict.items()
            }

    return {
        "by_year": {str(k): summarize(v) for k, v in sorted(by_year.items())},
        "by_category": {k: summarize(v) for k, v in sorted(by_category.items())},
        "by_diary": {k: summarize(v) for k, v in sorted(by_diary.items())},
        # 新增聚合
        "intensity_by_year": int_by_year,
        "emotion_by_year": emo_by_year,
    }


# ============================================================
# 6. 情感转折点检测
# ============================================================
def detect_turning_points(entries, scores, window=15):
    """
    检测情感转折点：滑动窗口中情感均值的显著变化
    返回按年份组织的重大转折点列表
    """
    dated = []
    for i, e in enumerate(entries):
        yr, mo, dy = e.get('year'), e.get('month'), e.get('day')
        if yr and mo and dy:
            try:
                from datetime import date as dt_date
                d = dt_date(yr, mo, dy)
                dated.append((d, i, scores[i]['score']))
            except ValueError:
                continue

    dated.sort(key=lambda x: x[0])

    if len(dated) < window * 2:
        return []

    scores_arr = np.array([s for _, _, s in dated])

    # 滑动窗口均值
    means = np.array([
        np.mean(scores_arr[i:i + window])
        for i in range(len(scores_arr) - window + 1)
    ])

    # 差分（后一窗口均值 - 前一窗口均值）
    diffs = np.diff(means)
    diff_std = np.std(diffs)
    threshold = diff_std * 0.5  # 相对阈值：超过半个标准差的差分视为显著
    if threshold < 0.02:
        threshold = 0.02  # 绝对下限

    turning_points = []
    for i in range(len(diffs)):
        if abs(diffs[i]) > threshold:
            mid_idx = i + window // 2
            tp_date = dated[mid_idx][0]
            # 局部极值：差分方向变化时
            is_extreme = False
            if i == 0:
                is_extreme = True
            elif i == len(diffs) - 1:
                is_extreme = True
            elif diffs[i] * diffs[i - 1] < 0:  # 方向改变
                is_extreme = True
            # 或者变化幅度极大（>2倍阈值）
            elif abs(diffs[i]) > threshold * 2:
                is_extreme = True

            if is_extreme:
                turning_points.append({
                    "date": tp_date.isoformat(),
                    "year": tp_date.year,
                    "change": round(float(diffs[i]), 4),
                    "type": "up" if diffs[i] > 0 else "down",
                    "mean_before": round(float(means[i]), 4),
                    "mean_after": round(float(means[i + 1]), 4),
                })

    # 同一年份只保留最明显的3个
    tps_by_year = {}
    for tp in turning_points:
        tps_by_year.setdefault(tp['year'], []).append(tp)

    result = []
    for year, tps in tps_by_year.items():
        tps.sort(key=lambda x: abs(x['change']), reverse=True)
        result.extend(tps[:3])

    result.sort(key=lambda x: x['date'])
    return result


# ============================================================
# 7. 主流程
# ============================================================
def analyze_all():
    print("=" * 60)
    print("  革命日记情感分析 v2.0 — 增强版")
    print("=" * 60)

    print("\n📚 加载条目元数据...")
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    print(f"   共 {len(meta)} 条日记")

    print("\n🔧 初始化分词器...")
    jieba = get_jieba()
    print(f"   📖 情感词典总量: {len(WORD_WEIGHT_MAP)} 词")
    print(f"      正向: {len(STRONG_POSITIVE)} 强 + {len(MEDIUM_POSITIVE)} 中 + {len(WEAK_POSITIVE)} 弱")
    print(f"      负向: {len(STRONG_NEGATIVE)} 强 + {len(MEDIUM_NEGATIVE)} 中 + {len(WEAK_NEGATIVE)} 弱")
    print(f"   🔄 否定词: {len(NEGATION_WORDS)} 个")
    print(f"   📊 程度副词: {len(INTENSIFIERS)} 个")
    print(f"   🎭 情感子类: {len(EMOTION_DICT)} 类")

    print("\n🔬 逐条分析情感...")
    entries_results = []
    total = len(meta)
    batch_size = 2000
    for i, e in enumerate(meta):
        text = e.get('text', '')
        result = score_entry(text, jieba)
        result['idx'] = i
        entries_results.append(result)

        if (i + 1) % batch_size == 0:
            pct = (i + 1) / total * 100
            print(f"   进度: {i+1}/{total} ({pct:.0f}%)")

    print(f"   完成: {total}/{total}")

    # 汇总统计
    scores_arr = np.array([r['score'] for r in entries_results])
    overall_stats = {
        "mean": round(float(np.mean(scores_arr)), 4),
        "std": round(float(np.std(scores_arr)), 4),
        "median": round(float(np.median(scores_arr)), 4),
        "min": round(float(np.min(scores_arr)), 4),
        "max": round(float(np.max(scores_arr)), 4),
        "positive_pct": round(float(np.mean(scores_arr > 0.05) * 100), 1),
        "negative_pct": round(float(np.mean(scores_arr < -0.05) * 100), 1),
        "neutral_pct": round(float(np.mean((scores_arr >= -0.05) & (scores_arr <= 0.05)) * 100), 1),
    }

    # 情感强度统计
    intensity_arr = np.array([r.get('intensity', 0) for r in entries_results])
    overall_stats["mean_intensity"] = round(float(np.mean(intensity_arr)), 4)
    overall_stats["high_intensity_pct"] = round(float(np.mean(intensity_arr > 0.3) * 100), 1)

    print(f"\n📊 总体统计:")
    print(f"   平均得分: {overall_stats['mean']}")
    print(f"   平均强度: {overall_stats['mean_intensity']}")
    print(f"   标准差:   {overall_stats['std']}")
    print(f"   正向:     {overall_stats['positive_pct']}%")
    print(f"   中立:     {overall_stats['neutral_pct']}%")
    print(f"   负向:     {overall_stats['negative_pct']}%")
    print(f"   高强度:   {overall_stats['high_intensity_pct']}%")

    print("\n📈 聚合统计...")
    emotion_profiles = [r.get('emotion_profile', {}) for r in entries_results]
    agg = aggregate(meta, entries_results, emotion_profiles)

    print("\n   按分类:")
    for cat, stats in agg['by_category'].items():
        if cat in ('未分类', ''):
            continue
        bar = '█' * max(1, int(stats['mean'] * 50 + 25))
        print(f"   {cat}: {stats['mean']:+.4f} {bar}")

    print("\n🔍 检测情感转折点...")
    turning_points = detect_turning_points(meta, entries_results)
    print(f"   发现 {len(turning_points)} 个情感转折点")

    # 组装输出
    output = {
        "total": total,
        "method": "enhanced_domain_lexicon_v2",
        "method_details": "三级强度领域词典 + 否定词处理 + 程度副词调整 + 情感子类分析",
        "lexicon_size": {
            "positive": len(POSITIVE_SET),
            "negative": len(NEGATIVE_SET),
            "total_weighted": len(WORD_WEIGHT_MAP),
            "negation_words": len(NEGATION_WORDS),
            "intensifiers": len(INTENSIFIERS),
            "emotion_categories": len(EMOTION_DICT),
        },
        "stats": overall_stats,
        "entries": entries_results,
        "turning_points": turning_points,
        **agg,
    }

    print(f"\n💾 保存结果...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"   已保存到 {OUTPUT_PATH} ({file_size:.1f}MB)")

    print(f"\n✅ 情感分析 v2.0 完成！")
    return output


if __name__ == "__main__":
    analyze_all()
