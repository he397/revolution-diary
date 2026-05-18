"""
革命日记语义检索系统 - 混合检索
===============================
方案：jieba 分词 + TF-IDF + 查询扩展（DeepSeek API）

优势：
  - 建索引只需几秒钟
  - TF-IDF 提供快速全文检索
  - DeepSeek API 将用户查询扩展为同义/相关词，弥补语义理解
  - 每次搜索只调 1 次 API，费用极低

使用：
  python3 build_search_index.py    # 构建索引（几秒完成）
  streamlit run search_ui.py        # 启动界面
"""

import os, json, pickle, re, hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# 配置
# ============================================================
DATA_FILE = os.path.join(BASE_DIR, "parsed_data/diaries_structured.json")
INDEX_DIR = os.path.join(BASE_DIR, "search_index")
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "tfidf_index.pkl")
META_PATH = os.path.join(INDEX_DIR, "entries_meta.pkl")
CLASSIFIED_FILE = os.path.join(BASE_DIR, "parsed_data/classified_entries.json")


# ============================================================
# jieba 分词器
# ============================================================
def get_jieba():
    import jieba
    # 加载领域词典，提高分词准确率
    domain_words = [
        "长征", "抗日", "解放战争", "八路军", "新四军", "红军", "革命",
        "南泥湾", "大生产", "陕甘宁", "晋察冀", "根据地",
        "宿营", "行军", "作战", "战斗", "突围", "歼灭",
        "指战员", "同志", "政委", "司令员", "旅长", "团长",
        "群众", "老百姓", "民夫", "老乡", "工作队",
        "土改", "整风", "大生产", "生产", "开荒", "种地",
        "机枪", "步枪", "手榴弹", "炮弹", "子弹",
        "合作社", "公粮", "救国公粮", "慰问", "劳军",
        "冬衣", "粮食", "伙食", "伤病员", "卫生队",
        "党支部", "党员", "支部会议", "组织生活",
        "俘虏", "伪军", "日军", "鬼子", "汉奸",
        "游击战", "运动战", "阵地战", "攻坚战",
    ]
    for w in domain_words:
        jieba.add_word(w)
    return jieba


def tokenize(text, jieba):
    """分词，返回空格分隔的词序列"""
    text = re.sub(r'[\s\n\r]+', ' ', text)
    words = jieba.lcut(text)
    # 去停用词（单字词、纯标点等）
    words = [w.strip() for w in words if len(w.strip()) > 1 and not re.match(r'^[，。、；：！？""''（）【】《》\\s\\d]+$', w)]
    return ' '.join(words)


# ============================================================
# 索引构建
# ============================================================
def prepare_entries():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = [e for e in data['entries'] if e['text'] and len(e['text']) > 5]
    return entries


def format_text(e):
    """格式化文本用于检索"""
    date_str = f"{e['year']}年" if e['year'] else ""
    if e['month'] and e['day']:
        date_str += f"{e['month']}月{e['day']}日"
    ws = f"天气:{e['weather']}" if e['weather'] else ""
    return f"{e['diary_name']} {date_str} {ws} {e['text'][:800]}"


def build_index():
    print("=" * 50)
    print("🔥 革命日记检索 - 构建索引 (TF-IDF + jieba)")
    print("=" * 50)

    entries = prepare_entries()
    print(f"\n📚 加载 {len(entries)} 条日记")

    # 将分类数据合并到条目中（按顺序匹配）
    if os.path.exists(CLASSIFIED_FILE):
        with open(CLASSIFIED_FILE, 'r', encoding='utf-8') as f:
            classified = json.load(f)
        classified_entries = classified if isinstance(classified, list) else classified.get('entries', [])
        matched = 0
        for i, e in enumerate(entries):
            if i < len(classified_entries):
                ce = classified_entries[i]
                cat = ce.get('category', '')
                e['category'] = cat if cat and cat != '未分类' else ''
                e['sub_tag'] = ce.get('sub_tag', '') if e.get('category') else ''
                if e['category']:
                    matched += 1
            else:
                e['category'] = ''
                e['sub_tag'] = ''
        print(f"   📋 已合并分类信息: {matched}/{len(entries)} 条有分类")
    else:
        for e in entries:
            e['category'] = ''
            e['sub_tag'] = ''

    print("🔨 分词...")
    jieba = get_jieba()
    texts = [format_text(e) for e in entries]

    # 分词（分批显示进度）
    tokenized = []
    for i, t in enumerate(texts):
        tokenized.append(tokenize(t, jieba))
        if (i + 1) % 5000 == 0:
            print(f"   分词进度: {i+1}/{len(texts)}")
    print(f"   分词完成: {len(tokenized)} 条")

    print(f"🤖 训练 TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(tokenized)
    print(f"   词表大小: {len(vectorizer.get_feature_names_out())}")
    print(f"   矩阵形状: {tfidf_matrix.shape}")

    # 保存
    print(f"💾 保存索引...")
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'tokenized_texts': tokenized,
        }, f)
    with open(META_PATH, 'wb') as f:
        pickle.dump(entries, f)

    print(f"\n✅ 索引构建完成！")
    print(f"   索引: {INDEX_PATH}  ({os.path.getsize(INDEX_PATH)/1024/1024:.0f}MB)")
    print(f"   条目数: {len(entries)}")
    print(f"   词表: {len(vectorizer.get_feature_names_out())} 词")


# ============================================================
# 搜索
# ============================================================
class SearchEngine:
    """混合搜索引擎：TF-IDF + DeepSeek 查询扩展"""

    def __init__(self):
        with open(INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
        with open(META_PATH, 'rb') as f:
            self.meta = pickle.load(f)
        self.jieba = get_jieba()
        self.api_key = self._get_api_key()

    def get_categories(self):
        """获取所有可用大类"""
        cats = set()
        for e in self.meta:
            if e.get('category'):
                cats.add(e['category'])
        return sorted(cats)

    def get_sub_tags(self, category):
        """获取某个大类下的所有子标签"""
        tags = set()
        for e in self.meta:
            if e.get('category') == category and e.get('sub_tag'):
                tags.add(e['sub_tag'])
        return sorted(tags)

    def _get_api_key(self):
        kf = os.path.join(INDEX_DIR, "api_key.txt")
        if os.path.exists(kf):
            with open(kf) as f:
                return f.read().strip()
        return ""

    def save_api_key(self, key):
        with open(os.path.join(INDEX_DIR, "api_key.txt"), 'w') as f:
            f.write(key)
        self.api_key = key

    def _expand_query(self, query):
        """用 DeepSeek API 扩展查询，提升语义理解"""
        if not self.api_key:
            return [query]

        import requests
        prompt = (
            f"用户正在搜索「革命日记」语料库。请将以下查询扩展为5-10个相关的搜索词（同义词、近义词、相关概念），"
            f"每行一个，只输出关键词不要序号和说明。\n\n查询词：{query}"
        )
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=15
            )
            if resp.status_code == 200:
                expanded = resp.json()["choices"][0]["message"]["content"].strip().split('\n')
                expanded = [q.strip() for q in expanded if q.strip()]
                all_queries = [query] + expanded
                print(f"  查询扩展: {all_queries}")
                return all_queries
        except Exception as e:
            pass
        return [query]

    def search(self, query, top_k=15, min_year=None, max_year=None, diaries=None,
               use_expansion=True, category=None, sub_tag=None):
        """搜索日记"""
        queries = self._expand_query(query) if use_expansion and self.api_key else [query]

        all_scores = None
        for q in queries:
            q_vec = self.vectorizer.transform([tokenize(q, self.jieba)])
            scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            if all_scores is None:
                all_scores = scores
            else:
                all_scores = np.maximum(all_scores, scores)

        top_indices = all_scores.argsort()[::-1]

        results = []
        for idx in top_indices:
            score = float(all_scores[idx])
            if score < 0.01:
                continue

            e = dict(self.meta[idx])
            e['score'] = score

            # 分类过滤
            if category:
                if not e.get('category') or e['category'] != category:
                    continue
                if sub_tag and e.get('sub_tag') != sub_tag:
                    continue

            results.append(e)

        # 非分类筛选（年代、日记名）
        filtered = []
        for r in results:
            if min_year and (r['year'] is None or r['year'] < min_year): continue
            if max_year and (r['year'] is None or r['year'] > max_year): continue
            if diaries and r['diary_name'] not in diaries: continue
            filtered.append(r)
            if len(filtered) >= top_k: break

        return filtered

    def browse(self, top_k=500, min_year=None, max_year=None, diaries=None,
               category=None, sub_tag=None):
        """浏览模式：不搜索，直接按分类/筛选条件返回条目（按日期排序）"""
        matched = []
        for idx in range(len(self.meta)):
            e = self.meta[idx]

            if category:
                if not e.get('category') or e['category'] != category:
                    continue
                if sub_tag and e.get('sub_tag') != sub_tag:
                    continue

            if min_year and (e['year'] is None or e['year'] < min_year): continue
            if max_year and (e['year'] is None or e['year'] > max_year): continue
            if diaries and e['diary_name'] not in diaries: continue

            matched.append(idx)

        # 按日期排序，取前 top_k 条
        def sort_key(idx):
            e = self.meta[idx]
            return (e['year'] or 9999, e['month'] or 99, e['day'] or 99)

        matched.sort(key=sort_key)
        if top_k:
            matched = matched[:top_k]

        results = []
        for idx in matched:
            e = dict(self.meta[idx])
            e['score'] = 1.0
            results.append(e)

        return results


if __name__ == "__main__":
    build_index()
