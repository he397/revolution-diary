"""
批量分类脚本 - 将日记条目归类到六大类+细化子标签
===============================================
使用 DeepSeek API 自动分类。

分类体系（六大类 + 细化子标签）：
  日常生活: 住宿/行军宿营, 伙食/饮食, 文娱活动, 穿着/衣物,
            伤病/医疗, 购物/贸易, 家务/杂务, 书信/通讯, 天气/气候, 其他
  军事作战: 行军/转移, 战斗/作战, 训练/演习, 侦察/情报,
            武器装备, 后勤/补给, 站岗/警戒, 伤亡/战果, 战略/部署, 其他
  组织建设: 会议/决议, 学习/培训, 发展党员, 组织生活,
            整风/整顿, 干部/人事, 批评/自我批评, 报告/总结, 其他
  群众运动: 减租减息, 支前/劳军, 土地改革, 妇女工作,
            青年工作, 农民运动, 工人运动, 宣传/动员, 拥政爱民, 其他
  政权建设: 选举/民主, 税收/公粮, 司法/锄奸, 行政管理,
            政策/法令, 经济/生产, 教育/办学, 统战/外交, 其他
  文化建设: 报刊/出版, 宣传/标语, 戏剧/文艺, 歌咏/音乐,
            教育/学习, 墙报/画报, 写作/创作, 图书/阅览, 其他

输出：parsed_data/classified_entries.json
"""

import os, json, requests, time, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "parsed_data/diaries_structured.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "parsed_data/classified_entries.json")
API_KEY_FILE = os.path.join(BASE_DIR, "search_index/api_key.txt")

CATEGORIES = {
    "日常生活": ["住宿/行军宿营", "伙食/饮食", "文娱活动", "穿着/衣物",
                 "伤病/医疗", "购物/贸易", "家务/杂务", "书信/通讯", "天气/气候", "其他"],
    "军事作战": ["行军/转移", "战斗/作战", "训练/演习", "侦察/情报",
                 "武器装备", "后勤/补给", "站岗/警戒", "伤亡/战果", "战略/部署", "其他"],
    "组织建设": ["会议/决议", "学习/培训", "发展党员", "组织生活",
                 "整风/整顿", "干部/人事", "批评/自我批评", "报告/总结", "其他"],
    "群众运动": ["减租减息", "支前/劳军", "土地改革", "妇女工作",
                 "青年工作", "农民运动", "工人运动", "宣传/动员", "拥政爱民", "其他"],
    "政权建设": ["选举/民主", "税收/公粮", "司法/锄奸", "行政管理",
                 "政策/法令", "经济/生产", "教育/办学", "统战/外交", "其他"],
    "文化建设": ["报刊/出版", "宣传/标语", "戏剧/文艺", "歌咏/音乐",
                 "教育/学习", "墙报/画报", "写作/创作", "图书/阅览", "其他"],
}
CAT_NAMES = list(CATEGORIES.keys())
# 所有子标签
ALL_SUBTAGS = set()
for tags in CATEGORIES.values():
    for t in tags:
        ALL_SUBTAGS.add(t)


def get_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()
    return os.environ.get("DEEPSEEK_API_KEY", "")


def classify_batch(entries_batch, api_key):
    """用 DeepSeek 批量分类一批条目"""
    prompt = """你是一位专业的历史文献分类专家。请为以下每条日记内容判断所属类别和子标签。

分类体系：
"""
    for cat, tags in CATEGORIES.items():
        prompt += f"- {cat}（子标签：{'、'.join(tags)}）\n"
    prompt += """
注意：
- 认真阅读每条内容，选择最匹配的大类和最具体的子标签
- 如果内容涉及多个方面，选最主要的一个
- 实在无法归入六大类的，大类用「其他」，子标签用「其他」
- 每行只输出「大类-子标签」格式，不要序号，不要多余文字
- 输出必须严格使用上面列出的类别和子标签名称

待分类条目：
"""
    for i, e in enumerate(entries_batch):
        text = e['text'][:300].replace('\n', ' ')
        date_info = ""
        if e.get('year'):
            date_info = f"{e['year']}年"
        if e.get('month') and e.get('day'):
            date_info += f"{e['month']}月{e['day']}日"
        if date_info:
            prompt += f"{i+1}. [{date_info}] {text}\n"
        else:
            prompt += f"{i+1}. {text}\n"

    for retry in range(3):
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.01,
                    "max_tokens": min(len(entries_batch) * 25, 4096)
                },
                timeout=60
            )
            if resp.status_code == 200:
                text = resp.text
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    time.sleep(1)
                    continue
                lines = [l.strip() for l in content.split('\n') if '-' in l and l.strip()]
                results = []
                for l in lines:
                    l_clean = re.sub(r'^\d+[.、．\s]+', '', l).strip()
                    parts = l_clean.split('-', 1)
                    if len(parts) == 2:
                        cat, tag = parts[0].strip(), parts[1].strip()
                        if cat in CAT_NAMES and tag in ALL_SUBTAGS:
                            results.append((cat, tag))
                            continue
                    results.append(("其他", "其他"))
                results = results[:len(entries_batch)]
                while len(results) < len(entries_batch):
                    results.append(("其他", "其他"))
                return results
            else:
                time.sleep(1)
        except requests.exceptions.Timeout:
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return [("其他", "其他")] * len(entries_batch)


def main():
    print("=" * 50)
    print("📂 日记条目自动分类（细化版）")
    print("=" * 50)

    api_key = get_api_key()
    if not api_key:
        print("❌ 未找到 API Key")
        print(f"   请将 API Key 存入 {API_KEY_FILE}")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = [e for e in data['entries'] if e['text'] and len(e['text']) > 5]
    print(f"📚 待分类条目: {len(entries)} 条")

    batch_size = 150
    total = len(entries)
    classified = 0

    for i in range(0, total, batch_size):
        batch = entries[i:i+batch_size]
        results = classify_batch(batch, api_key)
        for j, (cat, tag) in enumerate(results):
            batch[j]['category'] = cat
            batch[j]['sub_tag'] = tag
        classified += len(batch)
        pct = classified / total * 100
        bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
        print(f"   [{bar}] {classified}/{total} ({pct:.0f}%)")
        time.sleep(0.3)

    cat_counts = {}
    for e in entries:
        c = e.get('category', '其他')
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"\n📊 分类统计:")
    for c in CAT_NAMES + ["其他"]:
        cnt = cat_counts.get(c, 0)
        bar = '█' * max(1, cnt // 200)
        print(f"   {c}: {cnt:5d}条 {bar}")

    output = {
        "total": len(entries),
        "classification_stats": cat_counts,
        "entries": entries
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
