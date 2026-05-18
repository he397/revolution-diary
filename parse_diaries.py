"""
革命日记语料库 - 数据解析脚本 v3 (多文件版)
========================================
将「抗战日记（44本清洗版）」文件夹中的独立txt文件解析为结构化的日记条目数据。

输入：抗战日记（44本清洗版）/ —— 44本日记的独立txt文件
输出：
  - diaries_structured.json   → 完整结构化数据（供程序使用）
  - diaries_overview.csv      → 概览表格（供人工检查）
  - diaries_by_author/        → 按作者拆分的子文件
"""

import re
import json
import os
import csv
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# 配置
# ============================================================
DIARY_DIR = os.path.join(BASE_DIR, "抗战日记（43本清洗版）")
OUTPUT_DIR = os.path.join(BASE_DIR, "parsed_data")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "diaries_structured.json")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "diaries_overview.csv")
AUTHOR_DIR = os.path.join(OUTPUT_DIR, "by_author")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUTHOR_DIR, exist_ok=True)


# ============================================================
# 第一步：按日记名称分割
# ============================================================
DIARY_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d+\.\s*《[^》]+》[^\n]*)'
)

def split_diaries(text):
    """将全文按日记名称分割为独立的日记块"""
    positions = []
    for m in DIARY_PATTERN.finditer(text):
        positions.append((m.start(), m.group(1).strip()))

    blocks = []
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start:end]
        blocks.append({
            "name": name,
            "raw_text": content,
            "char_count": len(content)
        })
    return blocks


# ============================================================
# 第二步：分离前言与正文
# ============================================================
def split_intro_vs_entries(diary_name, raw_text):
    """
    每本日记的结构通常是：
      【前言部分】编者的话、照片说明、目录等
      【正文部分】日期+日记内容
    通过检测第一个日期出现的位置来分割。
    """
    # 更丰富的日期模式
    date_patterns = [
        r'(?:^|\n)\s*[一二三四五六七八九十\d]{1,4}月[一二三四五六七八九十\d]{1,4}日',  # 三月一日/3月5日
        r'(?:^|\n)\s*\d{1,2}\s*月\s*\d{1,2}\s*日',  # 3 月 5 日（有空格）
        r'(?:^|\n)\s*(?:19[2-5]\d年.*?(?:\d{1,2}月(?:\d{1,2}日)?)?)',  # 1949年
        r'(?:^|\n)\s*[一二三四五六七八九十]{1,4}月',  # 三月（月份标题）
        r'(?:^|\n)\s*19[2-5]\d[.,，、]\d{1,2}[.,，、]\d{1,2}',  # 1942.9.21
        r'(?:^|\n)\s*[一九二三四五六七八九十]+\s*年(?!月)',  # 一九三七年（不带月份）
    ]

    # 跳过关键词（前言/目录/插图等区域的标识）
    skip_keywords = [
        '编者', '前言', '目录', '说明', '照片', '手迹', '题词',
        '编辑', '出版', '领导小组', '回忆', '纪念', '插图',
        '序', '跋', '后记', '作者简介', '内容提要', '凡例',
        '责编', '封面', '扉页', '封底', '版权', '印数', '定价',
        '新华出版社', '人民出版社', '文献出版社', '印刷',
    ]

    lines = raw_text.split('\n')
    # 去掉第一行（标题行），但保留日期行
    content_lines = lines
    if lines:
        first = lines[0].strip()
        # 如果第一行是日期/年份行（如"一九二六年"、"1949年"），则保留
        is_year = bool(re.match(r'^(?:19[2-5]\d年|[一九二三四五六七八九十]+\s*年)', first))
        is_date_or_month = bool(re.match(r'^[一二三四五六七八九十\d]{1,4}月', first))
        is_title = not (is_year or is_date_or_month)
        content_lines = lines[1:] if is_title else lines

    first_date_idx = None
    first_date_str = None

    for i, line in enumerate(content_lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # 跳过前言区域关键词
        if any(kw in line_stripped for kw in skip_keywords):
            continue
        # 跳过页码行（纯数字）
        if line_stripped.isdigit():
            continue
        # 跳过目录中的点线行（如 "1949年．．．．．．"）
        if re.match(r'^\d{4}年[．\.]+', line_stripped):
            continue
        # 跳过照片说明行（但不要跳过纯年份行如"一九三七年"/"1949年"）
        if re.match(r'^[^\d\n]{0,10}(?:年|月|日)[^\d\n]{0,10}$', line_stripped) and len(line_stripped) < 30:
            # 纯年份行（中文或数字）不是照片说明，不跳过
            if re.match(r'^[一二三四五六七八九零〇\d\s]+\s*年$', line_stripped):
                pass
            # 纯月份行（如"七月"、"八月"）也不是照片说明，不跳过
            elif re.match(r'^[一二三四五六七八九十]{1,4}月$', line_stripped):
                pass
            else:
                continue

        for pat in date_patterns:
            if re.match(pat, line):
                first_date_idx = i
                first_date_str = line_stripped[:60]
                break
        if first_date_idx is not None:
            break

    if first_date_idx is None:
        return {"intro": raw_text, "entries_raw": "", "note": "未检测到日期条目"}

    intro_text = '\n'.join(content_lines[:first_date_idx]).strip()
    entries_text = '\n'.join(content_lines[first_date_idx:]).strip()

    return {"intro": intro_text, "entries_raw": entries_text, "note": "ok"}


# ============================================================
# 第三步：解析单条日记条目
# ============================================================
# 中文数字 → 阿拉伯数字
CN_MAP = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '廿': 20, '卅': 30,
}

def cn2num(s):
    """中文数字转阿拉伯数字"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s in CN_MAP:
        return CN_MAP[s]
    # 处理 "十二" = 12, "二十三" = 23, "二十一" = 21
    total = 0
    temp = 0
    for c in s:
        if c in CN_MAP:
            v = CN_MAP[c]
            if v >= 10:
                if temp == 0:
                    total += v
                else:
                    total += temp * v
                    temp = 0
            else:
                temp = v
    total += temp
    return total if total > 0 else 0


CN_NUM_CHARS = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10, '〇':0, '零':0}
def cn_year_to_num(s):
    """'一九三三' → 1933, '一九四九' → 1949"""
    result = 0
    for c in s:
        if c in CN_NUM_CHARS:
            result = result * 10 + CN_NUM_CHARS[c]
    return result if result > 1900 else None


# 日期正则
RE_YEAR = re.compile(r'^\s*(19[2-5]\d)\s*年\s*$')
RE_YEAR_CN = re.compile(r'^\s*([一九二三四五六七八九十]{4,6})\s*年\s*$')  # 一九三三年（整行仅年份）
RE_YEAR_CN_SPACED = re.compile(r'^\s*([一九二三四五六七八九十]+\s*[一九二三四五六七八九十]+)\s*年\s*$')  # 一九 三七 年
RE_DATE_FULL = re.compile(r'^\s*(19[2-5]\d)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日')  # 1949年1月2日...
RE_DATE_NUM = re.compile(r'^\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日')
RE_DATE_CN = re.compile(r'^\s*([一二三四五六七八九十廿卅\d]{1,4})月([一二三四五六七八九十廿卅\d]{1,4})日')
RE_DATE_NUM_CN = re.compile(r'^\s*(\d{1,2})\s*月\s*([一二三四五六七八九十廿卅\d]{1,4})日')
RE_DATE_DOT = re.compile(r'^\s*(19[2-5]\d)[.,，、](\d{1,2})[.,，、](\d{1,2})')  # 1942.9.21 / 1942,10,10
# 月份标题（独立成行）：七月、八月...
RE_MONTH_CN = re.compile(r'^\s*([一二三四五六七八九十]{1,4})\s*月\s*$')
# 纯日期（仅有日，不带月）：二十日，晴...
RE_DAY_CN = re.compile(r'^\s*([初一二三四五六七八九十廿卅\d]{1,4})日[，,、\s]*(.*)')

# 天气词
WEATHER_WORDS = ['晴', '阴', '雨', '雪', '风', '多云', '霜', '雾', '雷', '小雪', '大雨', '暴雨', '微风']

# 照片说明、手迹说明等跳过正则
SKIP_ENTRY_PATTERNS = [
    re.compile(r'手迹'),
    re.compile(r'摄于'),
    re.compile(r'照片'),
    re.compile(r'本色.*日记$'),
    re.compile(r'^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*$'),
    re.compile(r'^\s*[•·]\s*$'),
]


def parse_entries(diary_name, entries_text):
    if not entries_text or entries_text == "未检测到日期条目":
        return []

    entries = []
    lines = entries_text.split('\n')

    current_year = None
    current_date = None
    current_date_month = None
    current_date_day = None
    current_text_lines = []
    pending_month = None  # 月份标题设此值（如林伯渠:"七月"+"二十日"），待日期行使用

    def save_current():
        if current_date:
            text = '\n'.join(current_text_lines).strip()
            # 提取天气
            weather_found = ""
            paras = text.split('\n')
            first_line = paras[0].strip() if paras else ''
            for w in WEATHER_WORDS:
                if first_line == w or first_line.startswith(w) and len(first_line) <= 4:
                    weather_found = w
                    break
            # 去掉天气行
            clean_text = text
            if weather_found and len(first_line) <= 4:
                clean_text = '\n'.join(paras[1:]).strip()

            entries.append({
                "date_raw": current_date,
                "year": current_year if current_year else None,
                "month": current_date_month,
                "day": current_date_day,
                "text": clean_text,
                "weather": weather_found,
                "char_count": len(clean_text)
            })

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_text_lines:
                current_text_lines.append('')
            continue

        # 跳过照片说明、手迹说明等
        if any(p.search(stripped) for p in SKIP_ENTRY_PATTERNS):
            continue

        # 跳过单行目录（如 "1949年．．．．．．515"）
        if re.match(r'^\d{4}年[．\.]+', stripped):
            continue

        # 完整日期: "1949年1月2日千宿县晴"
        m_date_full = RE_DATE_FULL.match(line)
        if m_date_full:
            save_current()
            current_year = int(m_date_full.group(1))
            current_date_month = int(m_date_full.group(2))
            current_date_day = int(m_date_full.group(3))
            current_date = stripped[:60]
            rest = line[m_date_full.end():].strip()
            weather_found = ""
            for w in ['晴', '阴', '雨', '雪', '风', '多云', '霜', '雾', '雷']:
                if rest.endswith(w):
                    weather_found = w
                    rest = rest[:-len(w)].strip()
                    break
            current_text_lines = []
            if rest:
                current_text_lines.append(rest)
            continue

        # 点分日期: "1942.9.21" / "1942,10,10"
        m_date_dot = RE_DATE_DOT.match(line)
        if m_date_dot:
            save_current()
            current_year = int(m_date_dot.group(1))
            current_date_month = int(m_date_dot.group(2))
            current_date_day = int(m_date_dot.group(3))
            current_date = stripped[:60]
            rest = line[m_date_dot.end():].strip()
            current_text_lines = []
            if rest:
                current_text_lines.append(rest)
            continue

        # 年份行（数字）
        m_year = RE_YEAR.match(line)
        if m_year:
            save_current()
            current_year = int(m_year.group(1))
            continue

        # 年份行（中文：一九三三年）
        m_year_cn = RE_YEAR_CN.match(line)
        if m_year_cn:
            save_current()
            cn_y = cn_year_to_num(m_year_cn.group(1))
            if cn_y:
                current_year = cn_y
            continue

        # 年份行（中文带空格：一九 三七 年）
        m_year_cn_spaced = RE_YEAR_CN_SPACED.match(line)
        if m_year_cn_spaced:
            save_current()
            # 去掉空格后解析
            cn_str = re.sub(r'\s+', '', m_year_cn_spaced.group(1))
            cn_y = cn_year_to_num(cn_str)
            if cn_y:
                current_year = cn_y
            continue

        # 月份标题：七月（独立成行，后面日期只有日）
        m_month = RE_MONTH_CN.match(line)
        if m_month:
            mn = cn2num(m_month.group(1))
            if mn and 1 <= mn <= 12:
                pending_month = mn
                current_date_month = mn  # 同时设置，方便后续日条目使用
            continue

        # 数字日期: "3月5日"
        m_date = RE_DATE_NUM.match(line)
        if m_date:
            save_current()
            current_date = stripped
            current_date_month = int(m_date.group(1))
            current_date_day = int(m_date.group(2))
            current_text_lines = []
            continue

        # 中文日期: "三月一日"
        m_date_cn = RE_DATE_CN.match(line)
        if m_date_cn:
            save_current()
            current_date = stripped
            current_date_month = cn2num(m_date_cn.group(1))
            current_date_day = cn2num(m_date_cn.group(2))
            current_text_lines = []
            continue

        # 混合日期: "3月一日"
        m_mixed = RE_DATE_NUM_CN.match(line)
        if m_mixed:
            save_current()
            current_date = stripped
            current_date_month = int(m_mixed.group(1))
            current_date_day = cn2num(m_mixed.group(2))
            current_text_lines = []
            continue

        # 纯日条目（无月份，依赖 pending_month）："二十日，晴(星期二)"
        m_day_cn = RE_DAY_CN.match(line)
        if m_day_cn and pending_month is not None:
            day_val = cn2num(m_day_cn.group(1))
            if day_val and 1 <= day_val <= 31:
                save_current()
                current_date_day = day_val
                current_date_month = pending_month
                current_date = f"{pending_month}月{day_val}日"
                rest = m_day_cn.group(2).strip()
                # 提取紧跟在日之后的天气词
                weather_found = ""
                for w in ['晴', '阴', '雨', '雪', '风', '多云', '霜', '雾', '雷']:
                    if rest.startswith(w):
                        weather_found = w
                        rest = rest[len(w):].strip()
                        break
                current_text_lines = []
                if rest:
                    current_text_lines.append(rest)
                continue

        # 如果当前有活跃日期，累加文本
        if current_date:
            current_text_lines.append(stripped)

    # 保存最后一条
    save_current()

    # 推补年份
    infer_years(entries, diary_name)

    return entries


def infer_years(entries, diary_name):
    """为没有年份的条目推断年份"""
    # 先收集已知年份
    known = sorted(set(e['year'] for e in entries if e['year'] is not None))
    if not known:
        return

    # 获取日记名称中可能包含的年份范围
    name_years = re.findall(r'(19[2-5]\d)', diary_name)
    name_start = int(name_years[0]) if name_years else None

    # 逐条推补
    last_year = known[0]
    for e in entries:
        if e['year'] is not None:
            last_year = e['year']
        elif e['month'] is not None:
            # 月份序列推断
            e['year'] = last_year

    # 第二次遍历：修正跨年
    for i in range(1, len(entries)):
        prev = entries[i - 1]
        curr = entries[i]
        if prev['year'] and curr['year'] and curr['month'] and prev['month']:
            # 12月 → 1月，且年份相同 → 年份+1
            if prev['month'] == 12 and curr['month'] == 1 and curr['year'] == prev['year']:
                curr['year'] = prev['year'] + 1


# ============================================================
# 第四步：主流程
# ============================================================
def clean_diary_name(name):
    cleaned = re.sub(r'^\d+\.\s*', '', name)
    cleaned = re.sub(r'[「」""'']', '', cleaned)
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned


def generate_diary_id(name):
    m = re.search(r'《([^》]+)》', name)
    core = m.group(1) if m else name[:8]
    core = re.sub(r'\s+', '', core)
    return core[:6]


def get_diary_name_from_filename(filename):
    """从文件名提取日记名称"""
    name = filename
    # 去掉后缀
    name = re.sub(r'_(?:diary_only_)?副本\.txt$', '', name)
    name = re.sub(r'_formatted\.txt$', '', name)
    # 去掉前导编号，如 "1."、"20."
    name = re.sub(r'^\d+\.\s*', '', name)
    return name


def parse_all():
    print("=" * 60)
    print("革命日记语料库 - 数据解析 v2 (多文件版)")
    print("=" * 60)

    # 获取所有 txt 文件
    txt_files = sorted([f for f in os.listdir(DIARY_DIR) if f.endswith('.txt')])
    print(f"\n📖 发现 {len(txt_files)} 个日记文件")

    all_entries = []
    diary_summary = []
    total_chars = 0

    for idx, filename in enumerate(txt_files):
        filepath = os.path.join(DIARY_DIR, filename)
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        # 规范化：移除中文年月日数字之间的空格，如 "一九 三七 年" → "一九三七年"
        text = re.sub(r'([一九二三四五六七八九十])\s+(?=[一二三四五六七八九十]|年|月)', r'\1', text)

        name = get_diary_name_from_filename(filename)
        clean_name = clean_diary_name(name)
        diary_id = generate_diary_id(name)
        total_chars += len(text)

        print(f"\n  [{idx+1}/{len(txt_files)}] {clean_name[:40]}...")
        print(f"      大小: {len(text):,} 字符")

        split = split_intro_vs_entries(name, text)
        intro_len = len(split['intro'])

        entries = parse_entries(name, split['entries_raw'])

        print(f"      前言: {intro_len:,} 字符")
        print(f"      条目: {len(entries)} 条")

        years = sorted(set(e['year'] for e in entries if e['year']))
        year_range = f"{min(years)}-{max(years)}" if years else "未知"
        print(f"      年代: {year_range}")

        for e in entries:
            e['diary_id'] = diary_id
            e['diary_name'] = clean_name
            e['diary_index'] = idx + 1

        all_entries.extend(entries)

        diary_summary.append({
            "diary_id": diary_id,
            "diary_name": clean_name,
            "char_count": len(text),
            "entry_count": len(entries),
            "intro_len": intro_len,
            "year_range": year_range
        })

        # 按作者保存
        author_file = os.path.join(AUTHOR_DIR, f"{diary_id}.json")
        with open(author_file, 'w', encoding='utf-8') as f:
            json.dump({
                "diary_id": diary_id,
                "diary_name": clean_name,
                "char_count": len(text),
                "intro": split['intro'][:2000],
                "entries": entries,
                "entry_count": len(entries)
            }, f, ensure_ascii=False, indent=2)

    # 保存完整JSON
    print(f"\n💾 保存完整数据...")
    output = {
        "total_diaries": len(txt_files),
        "total_entries": len(all_entries),
        "total_chars": total_chars,
        "diaries": diary_summary,
        "entries": all_entries
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 保存CSV概览
    print(f"💾 保存CSV概览...")
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['编号', '日记ID', '日记名称', '日期原文', '年份', '月份', '日',
                        '天气', '内容摘要', '字数'])
        for idx, e in enumerate(all_entries, 1):
            summary = e['text'][:80].replace('\n', ' ') if e['text'] else ''
            writer.writerow([
                idx, e['diary_id'], e['diary_name'],
                e['date_raw'], e['year'], e['month'], e['day'],
                e['weather'], summary, e['char_count']
            ])

    # 统计输出
    print(f"\n{'=' * 60}")
    print(f"✅ 解析完成！")
    print(f"   日记总数: {len(txt_files)} 本")
    print(f"   条目总数: {len(all_entries):,} 条")
    print(f"   总字符数: {total_chars:,}")
    print(f"\n   输出文件:")
    print(f"     JSON: {OUTPUT_JSON}")
    print(f"     CSV:  {OUTPUT_CSV}")
    print(f"     按作者拆分: {AUTHOR_DIR}/")

    print(f"\n📊 各日记条目数:")
    for d in sorted(diary_summary, key=lambda x: x['entry_count'], reverse=True):
        bar = '█' * max(1, d['entry_count'] // 60)
        print(f"   {d['diary_id']:8s} | {d['entry_count']:5d}条 | {d['year_range']:12s} | {bar}")

    return output


if __name__ == "__main__":
    result = parse_all()
