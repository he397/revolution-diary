"""
跨日记事件详情
=============
同一事件在不同日记中的关联呈现，含情感影响、关联实体、原文
由 search_ui.py 导入使用
"""

import streamlit as st
import json
import os
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENT_DATA_PATH = os.path.join(BASE_DIR, "parsed_data/event_timeline.json")


@st.cache_data
def load_event_data(max_clusters=500, _mtime=0.0):
    """加载事件数据（_mtime 使缓存随文件修改自动失效）"""
    if not os.path.exists(EVENT_DATA_PATH):
        return None
    with open(EVENT_DATA_PATH) as f:
        data = json.load(f)
    data["event_clusters"] = data["event_clusters"][:max_clusters]
    for c in data["event_clusters"]:
        if len(c["mentions"]) > 10:
            c["mentions"] = c["mentions"][:10]
    return data


CATEGORIES = ["全部", "军事作战", "组织建设", "群众运动", "政权建设", "文化建设", "日常生活", "其他"]


def render_cross_diary_detail(clusters):
    """跨日记事件详情：列出跨日记事件，每条内联显示详情"""
    cross = [c for c in clusters if c["diary_count"] >= 2]
    if not cross:
        st.info("没有跨日记事件")
        return

    cross.sort(key=lambda x: (-x["diary_count"], -x["importance"]))

    for idx, c in enumerate(cross):
        # --- Category badge + name ---
        cat = c.get("category", "其他")
        cat_emoji = {
            "军事作战": "⚔️", "组织建设": "🏛️", "群众运动": "✊",
            "政权建设": "📜", "文化建设": "🎭", "日常生活": "🌾",
        }.get(cat, "📌")
        col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1, 1, 1])
        col1.markdown(f"**{c['name']}**　<small style='color:#888'>{cat_emoji} {cat}</small>",
                      unsafe_allow_html=True)
        col2.markdown(f"📅 {c['start_date'][:7]} ~ {c['end_date'][:7]}")
        col3.markdown(f"📓 **{c['diary_count']}** 本")
        col4.markdown(f"💬 {c['total_mentions']}")
        col5.markdown(f"⭐ {c['importance']:.2f}")

        # --- Sentiment impact ---
        si = c.get("sentiment_impact")
        if si and si.get("before_mean") is not None:
            s_col1, s_col2, s_col3 = st.columns([3, 3, 3])
            s_col1.metric("事件前 30 日均分", f"{si['before_mean']:.4f}",
                          help=f"样本数: {si.get('sample_before', 0)}")
            s_col2.metric("事件后 30 日均分", f"{si['after_mean']:.4f}",
                          help=f"样本数: {si.get('sample_after', 0)}")
            delta = si.get("change")
            if delta is not None:
                s_col3.metric("情感变化", f"{delta:+.4f}", delta=f"{delta:+.4f}")

        # --- Related entities ---
        re = c.get("related_entities", {})
        parts = []
        if re.get("persons"):
            parts.append(f"👤 {', '.join(re['persons'][:8])}")
        if re.get("locations"):
            parts.append(f"📍 {', '.join(re['locations'][:8])}")
        if re.get("organizations"):
            parts.append(f"🏛 {', '.join(re['organizations'][:5])}")
        if parts:
            st.markdown(" | ".join(parts))

        # --- Diary mentions grouped by diary ---
        diary_mentions = defaultdict(list)
        for m in c["mentions"]:
            diary_mentions[m["diary_name"]].append(m)

        d_items = list(diary_mentions.items())
        for di, (d_name, d_mentions) in enumerate(d_items):
            is_last = di == len(d_items) - 1
            branch = "└──" if is_last else "├──"
            expander_label = f"{branch} {d_name}（{len(d_mentions)}条）"
            with st.expander(expander_label, expanded=False):
                for m in d_mentions:
                    st.markdown(f"**{m['date']}**")
                    st.markdown(f"> {m['text_snippet']}")
                    st.markdown("")

        if idx < len(cross) - 1:
            st.divider()


# ====================================================================
# 关联词汇云
# ====================================================================

import jieba
from collections import Counter

CLASSIFIED_PATH = os.path.join(BASE_DIR, "parsed_data/classified_entries.json")

_STOPWORDS = frozenset({
    "的","了","在","是","我","有","和","就","不","人","都","一",
    "上","也","很","到","说","要","去","你","会","着",
    "没","看","好","这","他","她","它","们","又","再",
    "被","把","向","从","与","以","为","对","而","或",
    "过","让","给","比","等","之","所","将","于","其",
    "中","今","明","昨","此","哪","谁","各","每","某",
    "第","还","更","最","吧","吗","呀","呢","啊","哦",
    "哈","下","前","后","左","右","内","外","旁",
    "西","南","北","多","少","样","里",
    "出","入","进","回","能","当","同","便",
    "我们","他们","你们","自己","一个","没有","不是",
    "同志","工作","问题","因为","所以","但是","而且",
    "虽然","如果","已经","时候","还是","只是","就是",
    "可以","什么","怎么","这里","那里",
    "这个","那个","这些","那些","这样","那样",
    "今天","明天","昨天","上午","下午","晚上","准备",
    "知道","觉得","看见","听到","开始","继续",
    "许多","大量","很多","不少","主要","重要",
    "大家","一起","东西","情况","决定",
    "部分","方面","各种","全部","整个",
    "不能","可能","应该","关于",
    "敌人","学习","干部","群众","出来","回来","起来",
    # 日记高频常用词
    "今日","一日","一天","几天","数日","多日",
    "本日","当日","连日","即日",
    "上午","中午","午后","下午","傍晚","夜晚","夜间",
    "早晨","清晨","拂晓","黎明","黄昏","深夜",
    "从今","今后","将来","过去","从前",
    "这时","那时","此时","此刻","当时",
    "忽然","突然","猛然","逐渐","渐渐",
    "一直","始终","仍然","依然","仍旧",
    "本来","原来","原本","其实","实际",
    "真是","真是","实在","确实",
    "极其","十分","非常","比较","相当",
    "几乎","大约","大概","左右","上下",
    "每逢","每到","每当",
    "有点","有些","一些","一点",
    "全部","所有","一切",
    "为了","因为","由于","所以","于是",
    "但是","可是","然而","不过","只是",
    "如果","假如","倘若",
    "虽然","尽管","即使",
    "不但","不仅","不只",
    "而且","并且","况且",
    "或者","还是","要么",
    "之后","以后","以前","以来",
    "以内","以外","以上","以下",
    "的话","而言","来说",
    "是否","能否","可否",
    "必须","必要","务必",
    "可能","也许","或许",
    "应该","应当","该当",
    "能够","可以","允许",
    "看见","听见","感到","觉得",
    "知道","明白","了解","懂得",
    "希望","期望","盼望",
    "想了一想","想了想",
    "后来","然后","接着","随后",
    "眼前","面前","目前","当前",
    "一带","一带地方",
})

_PUNCT = frozenset("，。、；：？！（）【】——…·《》\"\"''‘’“” \n\t\r")


@st.cache_data
def load_classified(_mtime=0.0):
    """加载分类条目数据"""
    if not os.path.exists(CLASSIFIED_PATH):
        return None
    with open(CLASSIFIED_PATH) as f:
        return json.load(f)


@st.cache_data
def compute_word_freq(sub_tag, _mtime=0.0):
    """计算子标签下条目的词频"""
    classified = load_classified(_mtime)
    if not classified:
        return []

    texts = [e["text"] for e in classified["entries"] if e.get("sub_tag") == sub_tag]
    if not texts:
        return []

    all_words = []
    for text in texts:
        words = jieba.lcut(text)
        all_words.extend(
            w for w in words
            if len(w) >= 2
            and w not in _STOPWORDS
            and w not in _PUNCT
            and not w.isdigit()
            and w != sub_tag
        )

    counter = Counter(all_words)
    filtered = [(w, c) for w, c in counter.most_common(120)]
    return filtered[:60]


def render_knowledge_graph(sub_tag, word_freq):
    """渲染知识图谱（力导向图 — 半透明色彩）"""
    if not word_freq:
        st.info("该子标签暂无足够数据")
        return

    max_c = word_freq[0][1]
    min_c = word_freq[-1][1] if len(word_freq) > 1 else max_c
    range_c = max(max_c - min_c, 1)

    import json
    nodes = [{"id": sub_tag, "group": "center", "count": max_c}]
    links = []
    for word, count in word_freq[:50]:
        nodes.append({"id": word, "group": "leaf", "count": count})
        links.append({"source": sub_tag, "target": word, "value": count})
    graph_json = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False)

    html = f"""
    <div id="kg" style="width:100%;height:560px;background:linear-gradient(135deg,#f5f7fa 0%,#e8ecf1 100%);border-radius:12px;overflow:hidden;position:relative;"></div>
    <div style="text-align:right;color:#888;font-size:12px;margin-top:4px;">
        共 {len(word_freq)} 个关联词
    </div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    (function() {{
        var data = {graph_json};
        var container = document.getElementById('kg');
        var w = container.clientWidth;
        var h = 560;
        var svg = d3.select('#kg').append('svg')
            .attr('width', w).attr('height', h).style('background', 'transparent');
        var g = svg.append('g');
        svg.call(d3.zoom().scaleExtent([0.3, 4]).on('zoom', function(ev) {{
            g.attr('transform', ev.transform);
        }}));

        var maxCount = {max_c};
        var minCount = {min_c};
        var countRange = {range_c};

        // Color palette: vibrant with transparency-friendly tones
        var _colors = ['#E74C3C','#E67E22','#F1C40F','#1ABC9C','#3498DB'];
        function getColor(d, opacity) {{
            if (d.group === 'center') return 'rgba(192,57,43,' + (opacity || 1) + ')';
            var t = (d.count - minCount) / countRange;
            var idx = Math.min(Math.floor(t * 4), 3);
            var lt = (t * 4) - idx;
            var c = d3.color(d3.interpolateRgb(_colors[idx], _colors[idx + 1])(lt));
            c.opacity = opacity || 0.85;
            return c + '';
        }}

        function getR(d) {{
            if (d.group === 'center') return 26;
            var t = (d.count - minCount) / countRange;
            return Math.round(10 + t * 20);
        }}

        function getLinkDist(d) {{
            if (!d.value) return 180;
            var t = (d.value - minCount) / countRange;
            return Math.round(180 - t * 120);
        }}

        // Force simulation
        var simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(function(d) {{ return d.id; }})
                .distance(getLinkDist).strength(0.3))
            .force('charge', d3.forceManyBody().strength(-250))
            .force('center', d3.forceCenter(w / 2, h / 2))
            .force('collision', d3.forceCollide().radius(function(d) {{ return getR(d) + 18; }}));

        // Links
        var link = g.append('g').selectAll('line')
            .data(data.links).enter().append('line')
            .attr('stroke', function(d) {{
                var t = (d.value - minCount) / countRange;
                var idx = Math.min(Math.floor(t * 4), 3);
                var c = d3.color(d3.interpolateRgb(_colors[idx], _colors[idx + 1])((t * 4) - idx));
                c.opacity = 0.3;
                return c + '';
            }})
            .attr('stroke-width', function(d) {{
                return 1 + (d.value - minCount) / countRange * 2.5;
            }});

        // Nodes
        var node = g.append('g').selectAll('circle')
            .data(data.nodes).enter().append('circle')
            .attr('r', getR)
            .attr('fill', function(d) {{ return getColor(d, 0.85); }})
            .attr('stroke', function(d) {{
                if (d.group === 'center') return '#7B241C';
                return 'rgba(255,255,255,0.9)';
            }})
            .attr('stroke-width', 1.5)
            .attr('cursor', 'grab')
            .attr('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.12))')
            .call(d3.drag()
                .on('start', function(ev, d) {{
                    if (!ev.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                }})
                .on('drag', function(ev, d) {{ d.fx = ev.x; d.fy = ev.y; }})
                .on('end', function(ev, d) {{
                    if (!ev.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }}));

        // Labels
        g.append('g').selectAll('text')
            .data(data.nodes).enter().append('text')
            .text(function(d) {{ return d.id; }})
            .attr('x', function(d) {{ return d.x; }})
            .attr('y', function(d) {{ return d.y; }})
            .attr('font-size', function(d) {{
                var r = getR(d);
                return Math.max(10, Math.min(15, r * 0.55)) + 'px';
            }})
            .attr('fill', '#fff')
            .attr('paint-order', 'stroke')
            .attr('stroke', 'rgba(0,0,0,0.3)')
            .attr('stroke-width', 2.5)
            .attr('font-weight', 'bold')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('pointer-events', 'none')
            .attr('font-family', 'SimHei,STHeiti,Microsoft YaHei,sans-serif');

        // Tooltip
        var tooltip = d3.select('#kg').append('div')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.85)')
            .style('color', '#fff')
            .style('padding', '5px 12px')
            .style('border-radius', '6px')
            .style('font-size', '13px')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('z-index', 100)
            .style('font-family', 'SimHei,STHeiti,Microsoft YaHei,sans-serif')
            .style('white-space', 'nowrap');

        node.on('mouseover', function(ev, d) {{
            tooltip.style('opacity', 1)
                .html(d.group === 'center'
                    ? '<b>' + d.id + '</b>（中心节点）'
                    : '<b>' + d.id + '</b> — 共现 ' + d.count + ' 次');
            d3.select(this).attr('stroke', '#FFD700').attr('stroke-width', 2.5);
        }}).on('mousemove', function(ev) {{
            tooltip.style('left', (ev.offsetX + 14) + 'px')
                .style('top', (ev.offsetY - 12) + 'px');
        }}).on('mouseout', function() {{
            tooltip.style('opacity', 0);
            d3.select(this).attr('stroke', function(d) {{
                return d.group === 'center' ? '#7B241C' : 'rgba(255,255,255,0.9)';
            }}).attr('stroke-width', 1.5);
        }});

        simulation.on('tick', function() {{
            link.attr('x1', function(d) {{ return d.source.x; }})
                .attr('y1', function(d) {{ return d.source.y; }})
                .attr('x2', function(d) {{ return d.target.x; }})
                .attr('y2', function(d) {{ return d.target.y; }});
            node.attr('cx', function(d) {{ return d.x; }})
                .attr('cy', function(d) {{ return d.y; }})
                .attr('fill', function(d) {{ return getColor(d, 0.85); }});
            svg.selectAll('text').attr('x', function(d) {{ return d.x; }})
                .attr('y', function(d) {{ return d.y; }});
        }});
    }})();
    </script>
    """

    st.components.v1.html(html, height=600)
    st.caption("中心=子标签 | 连线越短越粗=共现越紧密 | 暖色=高频,冷色=低频 | 半透明气泡可叠加 | 可拖拽/滚轮缩放")
