import os
"""
革命日记检索 - 交互界面
启动: streamlit run search_ui.py
"""

import streamlit as st
import streamlit.components.v1 as components
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="革命日记全文检索", page_icon="📖", layout="wide")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_search_index import SearchEngine, build_index
from event_timeline_ui import load_event_data, render_cross_diary_detail, load_classified, compute_word_freq, render_knowledge_graph

CLASSIFIED_FILE = os.path.join(BASE_DIR, "parsed_data/classified_entries.json")

@st.cache_data
def load_stats_data():
    """加载统计数据（直接从 pickle 读取，避免 engine 缓存问题）"""
    import pandas as pd
    import pickle as _pickle
    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'rb') as f:
        meta = _pickle.load(f)
    records = []
    for e in meta:
        records.append({
            'diary_name': e.get('diary_name', ''),
            'year': e.get('year'),
            'month': e.get('month'),
            'day': e.get('day'),
            'category': e.get('category', '未分类') or '未分类',
            'sub_tag': e.get('sub_tag', '') or '',
            'text_len': len(e.get('text', '')),
        })
    return pd.DataFrame(records)


def get_clean_sub_tags(df):
    """去重后的子标签列表，合并'人事'重复项"""
    tags = sorted(df['sub_tag'].unique())
    tags = [t for t in tags if t]
    result = []
    seen = set()
    for t in tags:
        key = '人事（其他）' if '人事' in t else t
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def match_sub_tag(entry_tag, filter_tag):
    """子标签匹配（处理别名）"""
    if not filter_tag:
        return True
    if filter_tag == '人事（其他）' and entry_tag and '人事' in entry_tag:
        return True
    return entry_tag == filter_tag


# 革命根据地／重要地点坐标
LOCATION_COORDS = {
    # 革命圣地
    "延安": (36.5951, 109.4855), "井冈山": (26.5704, 114.1655),
    "瑞金": (25.8753, 116.0270), "遵义": (27.7254, 106.9272),
    "西柏坡": (38.3136, 114.0282), "陕甘宁": (36.5, 108.0),
    "晋察冀": (38.9, 114.5), "晋冀鲁豫": (36.5, 114.5),
    "晋绥": (38.0, 111.0), "南泥湾": (36.3, 109.5),
    "大别山": (31.2, 115.5), "太行山": (37.0, 114.0),
    "沂蒙山": (35.5, 118.0),
    "鄂豫皖": (31.5, 115.0), "湘鄂赣": (29.0, 114.0),
    "川陕": (32.5, 107.0),
    # 省市
    "北平": (39.9042, 116.4074), "北京": (39.9042, 116.4074),
    "南京": (32.0603, 118.7969), "上海": (31.2304, 121.4737),
    "天津": (39.3434, 117.3616), "重庆": (29.4316, 106.9123),
    "广州": (23.1291, 113.2644), "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398), "沈阳": (41.8057, 123.4315),
    "长春": (43.8961, 125.3236), "哈尔滨": (45.8038, 126.5350),
    "长沙": (28.2282, 112.9388), "南昌": (28.6829, 115.8582),
    "贵阳": (26.6470, 106.6302), "昆明": (25.0389, 102.7183),
    "成都": (30.5728, 104.0668), "兰州": (36.0611, 103.8343),
    "西宁": (36.6241, 101.7781), "银川": (38.4872, 106.2309),
    "呼和浩特": (40.8422, 111.7498), "乌鲁木齐": (43.8256, 87.6168),
    # 省份（近似中心）
    "黑龙江": (48.0, 128.0), "吉林": (43.7, 126.0),
    "辽宁": (41.5, 122.5), "热河": (41.5, 118.0),
    "察哈尔": (40.8, 114.9), "绥远": (40.8, 111.5),
    "宁夏": (37.0, 106.0), "甘肃": (36.0, 104.0),
    "陕西": (35.0, 109.0), "山西": (37.5, 112.0),
    "河北": (38.0, 116.0), "山东": (36.0, 118.0),
    "河南": (34.0, 114.0), "湖北": (31.0, 112.0),
    "湖南": (28.0, 112.0), "江西": (27.0, 116.0),
    "安徽": (32.0, 117.0), "江苏": (32.5, 120.0),
    "浙江": (29.0, 120.0), "福建": (26.0, 118.0),
    "广东": (23.5, 113.5), "广西": (23.0, 108.0),
    "贵州": (27.0, 107.0), "云南": (25.0, 102.0),
    "四川": (30.0, 104.0), "西康": (30.0, 102.0),
    "青海": (36.0, 96.0), "西藏": (31.0, 87.0),
    # 重要关隘／战场
    "卢沟桥": (39.8500, 116.2100), "台儿庄": (34.5600, 117.7300),
    "平型关": (39.3000, 114.0000), "娘子关": (37.9000, 113.8000),
    "山海关": (39.9900, 119.7500), "嘉峪关": (39.8000, 98.2700),
    "泸定桥": (29.9100, 102.2300), "安顺场": (29.2000, 102.3000),
    "腊子口": (34.1000, 104.0000), "娄山关": (28.0000, 107.0000),
    # 其他根据地／活动区
    "苏北": (34.0, 119.5), "苏南": (31.5, 120.0),
    "皖北": (33.0, 117.0), "皖南": (30.5, 118.0),
    "华中": (31.0, 115.0), "山东": (36.0, 118.0),
    "陕甘": (36.0, 108.0), "陕北": (37.0, 109.0),
}

LOCATION_PATTERN_CACHE = None


@st.cache_data
def get_tags_with_locations(diary_name):
    """返回指定日记中具有地点数据的子标签列表"""
    import pickle, re
    from collections import Counter

    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    loc_pattern = re.compile('|'.join(re.escape(l) for l in sorted(LOCATION_COORDS.keys(), key=len, reverse=True)))
    tags = Counter()
    for e in meta:
        if e.get('diary_name') != diary_name:
            continue
        text = e.get('text', '')
        if loc_pattern.search(text):
            t = e.get('sub_tag', '')
            if t:
                key = '人事（其他）' if '人事' in t else t
                tags[key] += 1
    return [t for t, _ in tags.most_common()]


@st.cache_data
def extract_route_data(diary_name, sub_tag):
    """提取单本日记指定子标签的按日期排序地点序列"""
    import pickle, re
    from collections import OrderedDict

    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    loc_pattern = re.compile('|'.join(re.escape(l) for l in sorted(LOCATION_COORDS.keys(), key=len, reverse=True)))

    entries = []
    for e in meta:
        if e.get('diary_name') != diary_name:
            continue
        if sub_tag and not match_sub_tag(e.get('sub_tag'), sub_tag):
            continue
        y, m, d = e.get('year'), e.get('month'), e.get('day')
        if not (y and m and d):
            continue
        text = e.get('text', '')
        locs = loc_pattern.findall(text)
        if locs:
            entries.append((y, m, d, locs[0]))

    if not entries:
        return None

    entries.sort(key=lambda x: (x[0], x[1], x[2]))

    route = OrderedDict()
    for y, m, d, loc in entries:
        if loc not in LOCATION_COORDS:
            continue
        key = (y, m, d)
        if key not in route:
            route[key] = {"date": f"{y:04d}-{m:02d}-{d:02d}", "location": loc,
                         "lat": LOCATION_COORDS[loc][0], "lon": LOCATION_COORDS[loc][1]}

    route_list = list(route.values())
    if len(route_list) < 2:
        return None
    return route_list


def split_route_into_trips(route_data, max_gap_days=180, diary_name=None):
    """将路线拆分为多段行程。

    红军长征日记地点密集 → 每15个地点为一段行程；
    其他日记 → 按时间间隔拆分（超过 max_gap_days 天则拆分）。
    """
    if not route_data or len(route_data) < 2:
        return [route_data] if route_data else []

    # 长征日记：每15个地点一段，避免线条重重叠叠
    if diary_name and '长征' in diary_name:
        trips = []
        for i in range(0, len(route_data), 15):
            trips.append(route_data[i:i + 15])
        return trips

    from datetime import datetime
    trips = []
    current = [route_data[0]]
    for i in range(1, len(route_data)):
        d1 = datetime.strptime(route_data[i-1]["date"], "%Y-%m-%d")
        d2 = datetime.strptime(route_data[i]["date"], "%Y-%m-%d")
        gap = (d2 - d1).days
        if gap > max_gap_days:
            trips.append(current)
            current = [route_data[i]]
        else:
            current.append(route_data[i])
    if current:
        trips.append(current)
    return trips


def _build_route_map(route_data):
    """生成 Leaflet HTML 路线地图（单段行程，贝塞尔曲线避免往返重叠）"""
    import json

    pts_json = json.dumps([{
        "lat": p["lat"], "lon": p["lon"],
        "loc": p["location"], "date": p["date"]
    } for p in route_data])

    segs_json = json.dumps([{
        "from_lat": route_data[i]["lat"], "from_lon": route_data[i]["lon"],
        "to_lat": route_data[i+1]["lat"], "to_lon": route_data[i+1]["lon"],
        "from_loc": route_data[i]["location"], "to_loc": route_data[i+1]["location"],
    } for i in range(len(route_data) - 1)])

    center_lat = sum(p["lat"] for p in route_data) / len(route_data)
    center_lon = sum(p["lon"] for p in route_data) / len(route_data)

    return f"""
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin:0; padding:0; }}
            #map {{ width:100%; height:480px; }}
            .loc-label {{
                background: rgba(255,255,255,0.9) !important;
                border: 1px solid #999 !important;
                border-radius: 3px !important;
                padding: 2px 6px !important;
                font-size: 13px !important;
                font-weight: bold !important;
                color: #333 !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
                white-space: nowrap !important;
            }}
            .loc-label::before {{
                border-top-color: rgba(255,255,255,0.9) !important;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
        var map = L.map('map', {{ zoomControl: true, attributionControl: false }})
            .setView([{center_lat}, {center_lon}], 4);
        L.tileLayer('https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}', {{
            maxZoom: 18, attribution: ''
        }}).addTo(map);

        var points = {pts_json};
        var segments = {segs_json};
        var totalSegs = segments.length;
        var hue = 210;

        // ---- 统计往返次数，用于贝塞尔偏移 ----
        var routeCount = {{}};
        segments.forEach(function(seg) {{
            var k = [seg.from_loc, seg.to_loc].sort().join('\\x00');
            routeCount[k] = (routeCount[k] || 0) + 1;
        }});
        var routeUsed = {{}};

        // ---- 逐段绘制贝塞尔曲线 ----
        segments.forEach(function(seg, i) {{
            var ratio = totalSegs > 1 ? i / (totalSegs - 1) : 0.5;
            var lightness = 78 - ratio * 48;
            var color = 'hsl(' + hue + ', 65%, ' + lightness + '%)';

            // 贝塞尔偏移：往返地点错开
            var rk = [seg.from_loc, seg.to_loc].sort().join('\\x00');
            routeUsed[rk] = (routeUsed[rk] || 0) + 1;
            var total = routeCount[rk] || 1;
            var idx = routeUsed[rk];
            var offset = (idx - (total + 1) / 2) * 0.3;

            var dx = seg.to_lon - seg.from_lon;
            var dy = seg.to_lat - seg.from_lat;
            var len = Math.sqrt(dx * dx + dy * dy);
            if (len < 0.001) return;
            var px = -dy / len, py = dx / len;

            var midLat = (seg.from_lat + seg.to_lat) / 2 + py * offset;
            var midLon = (seg.from_lon + seg.to_lon) / 2 + px * offset;

            var curvePts = [];
            var steps = 30;
            for (var s = 0; s <= steps; s++) {{
                var t = s / steps, mt = 1 - t;
                var lat = mt * mt * seg.from_lat + 2 * mt * t * midLat + t * t * seg.to_lat;
                var lon = mt * mt * seg.from_lon + 2 * mt * t * midLon + t * t * seg.to_lon;
                curvePts.push([lat, lon]);
            }}
            L.polyline(curvePts, {{
                color: color, weight: 4, opacity: 0.85
            }}).addTo(map);

            // 方向箭头（40% 处切线方向）
            var t = 0.4, mt = 1 - t;
            var arrowLat = mt * mt * seg.from_lat + 2 * mt * t * midLat + t * t * seg.to_lat;
            var arrowLon = mt * mt * seg.from_lon + 2 * mt * t * midLon + t * t * seg.to_lon;
            var tangLat = 2 * mt * (midLat - seg.from_lat) + 2 * t * (seg.to_lat - midLat);
            var tangLon = 2 * mt * (midLon - seg.from_lon) + 2 * t * (seg.to_lon - midLon);
            var angle = Math.atan2(tangLat, tangLon) * 180 / Math.PI;
            var arrowIcon = L.divIcon({{
                className: '',
                html: '<div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid ' + color + ';transform:rotate(' + (90 - angle) + 'deg);"></div>',
                iconSize: [12, 12],
                iconAnchor: [6, 6],
            }});
            L.marker([arrowLat, arrowLon], {{ icon: arrowIcon, interactive: false }}).addTo(map);
        }});

        // ---- 站点标记 ----
        var n = points.length;
        points.forEach(function(p, i) {{
            var ratio = n > 1 ? i / (n - 1) : 0.5;
            var r = Math.round(50 + ratio * 170);
            var g = Math.round(120 + ratio * 40);
            var b = Math.round(210 - ratio * 110);
            var color = 'rgb(' + r + ',' + g + ',' + b + ')';
            var radius = 6 + ratio * 3;
            var marker = L.circleMarker([p.lat, p.lon], {{
                radius: radius, fillColor: color, color: '#fff',
                weight: 2, fillOpacity: 0.9
            }}).addTo(map);
            marker.bindTooltip(p.loc, {{
                direction: 'top', offset: [0, -8], permanent: true,
                className: 'loc-label'
            }});
            marker.bindPopup('<b>' + p.loc + '</b><br/>' + p.date);
        }});

        map.fitBounds(L.latLngBounds(points.map(function(p) {{ return [p.lat, p.lon]; }})).pad(0.1));
        </script>
    </body>
    </html>
    """


def _build_point_map(loc_data):
    """生成 Leaflet HTML 地点分布图（红色标记，<10个永久显示标签，否则悬停显示）"""
    import json

    pts_json = json.dumps([{
        "lat": p["lat"], "lon": p["lon"],
        "loc": p["location"], "count": p["count"],
        "date_min": p.get("date_min", ""),
        "date_max": p.get("date_max", ""),
    } for p in loc_data])

    center_lat = sum(p["lat"] for p in loc_data) / len(loc_data)
    center_lon = sum(p["lon"] for p in loc_data) / len(loc_data)
    show_permanent = len(loc_data) < 10

    return f"""
    <html>
    <head>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin:0; padding:0; }}
            #map {{ width:100%; height:480px; }}
            .loc-label {{
                background: rgba(220,50,50,0.85) !important;
                border: 1px solid #a00 !important;
                border-radius: 3px !important;
                padding: 2px 6px !important;
                font-size: 13px !important;
                font-weight: bold !important;
                color: #fff !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
                white-space: nowrap !important;
            }}
            .loc-label::before {{
                border-top-color: rgba(220,50,50,0.85) !important;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
        var map = L.map('map', {{ zoomControl: true, attributionControl: false }})
            .setView([{center_lat}, {center_lon}], 4);
        L.tileLayer('https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}', {{
            maxZoom: 18, attribution: ''
        }}).addTo(map);

        var points = {pts_json};
        var maxCount = Math.max(...points.map(function(p) {{ return p.count; }}), 1);

        points.forEach(function(p) {{
            var radius = 6 + (p.count / maxCount) * 14;
            var marker = L.circleMarker([p.lat, p.lon], {{
                radius: radius,
                fillColor: '#e03030',
                color: '#fff',
                weight: 2,
                fillOpacity: 0.85
            }}).addTo(map);
            var dateInfo = '';
            if (p.date_min && p.date_max) {{
                dateInfo = (p.date_min == p.date_max) ? ' ' + p.date_min : ' ' + p.date_min + ' ~ ' + p.date_max;
            }}
            marker.bindTooltip(p.loc + '（' + p.count + '次）' + dateInfo, {{
                direction: 'top', offset: [0, -8],
                permanent: {'true' if show_permanent else 'false'},
                className: 'loc-label'
            }});
            var popupHtml = '<b>' + p.loc + '</b><br/>提及 ' + p.count + ' 次';
            if (p.date_min && p.date_max) {{
                popupHtml += '<br/>' + (p.date_min == p.date_max ? p.date_min : p.date_min + ' ~ ' + p.date_max);
            }}
            marker.bindPopup(popupHtml);
        }});

        map.fitBounds(L.latLngBounds(points.map(function(p) {{ return [p.lat, p.lon]; }})).pad(0.1));
        </script>
    </body>
    </html>
    """


@st.cache_data
def extract_diary_locations(sub_tag=None, year_range=None, diary_name=None):
    """提取指定条件下的日记地点分布，返回 list[lat, lon, location, count, date_min, date_max]"""
    import pickle, re
    from collections import defaultdict

    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Build location regex
    loc_pattern = re.compile('|'.join(re.escape(l) for l in sorted(LOCATION_COORDS.keys(), key=len, reverse=True)))

    loc_info = defaultdict(lambda: {"count": 0, "date_min": None, "date_max": None})
    for e in meta:
        if sub_tag and not match_sub_tag(e.get('sub_tag'), sub_tag):
            continue
        if year_range:
            y = e.get('year')
            if y is None:
                continue
            y_min, y_max = year_range
            if y_min and y < y_min:
                continue
            if y_max and y > y_max:
                continue
        if diary_name and e.get('diary_name') != diary_name:
            continue
        text = e.get('text', '')
        for m in loc_pattern.finditer(text):
            loc = m.group()
            if loc not in LOCATION_COORDS:
                continue
            loc_info[loc]["count"] += 1
            y, mth, d = e.get('year'), e.get('month'), e.get('day')
            if y and mth and d:
                date_str = f"{y:04d}-{mth:02d}-{d:02d}"
                cur = loc_info[loc]
                if cur["date_min"] is None or date_str < cur["date_min"]:
                    cur["date_min"] = date_str
                if cur["date_max"] is None or date_str > cur["date_max"]:
                    cur["date_max"] = date_str

    if not loc_info:
        return None

    rows = []
    for loc, info in sorted(loc_info.items(), key=lambda x: -x[1]["count"]):
        lat, lon = LOCATION_COORDS[loc]
        rows.append({
            "location": loc, "lat": lat, "lon": lon,
            "count": info["count"],
            "date_min": info["date_min"],
            "date_max": info["date_max"],
        })
    return rows
@st.cache_resource
def get_engine():
    try:
        return SearchEngine()
    except FileNotFoundError:
        st.error("⚠️ 索引未构建")
        return None

@st.cache_data
def get_meta():
    import json
    with open(os.path.join(BASE_DIR, "parsed_data/diaries_structured.json")) as f:
        data = json.load(f)
    entries = data['entries']
    diaries = sorted(set(e['diary_name'] for e in entries if e.get('diary_name')))
    years = sorted(set(e['year'] for e in entries if e.get('year')))
    return diaries, years

@st.cache_data
def get_categories():
    engine = get_engine()
    if engine:
        return engine.get_categories()
    return []

@st.cache_data
def get_diary_list():
    """获取所有日记名称列表"""
    import pickle as _pickle
    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = _pickle.load(f)
    diaries = sorted(set(e.get('diary_name', '') for e in meta if e.get('diary_name')))
    return diaries

def _clean_name(name):
    """标准化文件名：去元数据后缀，保留卷号/上下等区分信息"""
    import re
    # 去掉括号内的各种元数据
    meta_patterns = [
        r'z[-.]library', r'\dlib\.sk', r'作者', r'著\b', r'整理', r'编\b',
        r'中国人民解放军', r'解放军', r'档案馆', r'出版社', r'出版',
        r'印刷', r'印数', r'定价', r'ISBN', r'可上传',
        r'\d{10,}', r'_\d{6,}',
    ]
    for pat in meta_patterns:
        name = re.sub(f'[（(][^）)]*{pat}[^）)]*[）)]', '', name)
    # 去掉括号内含年份范围如 （1939-1945）
    name = re.sub(r'[（(]\d{4}[-–]\d{4}[^）)]*[）)]', '', name)
    # 先提取带卷号标记的括号内容（保留下来用于匹配）
    vol_keep = re.findall(r'[（(][^）)]*[卷册部][^）)]*[）)]', name)
    # 去掉括号内纯数字/字母如 (上、下册) (1)
    name = re.sub(r'[（(][^）)]*[上下中]?[、,，]?[册部卷]?[）)]', '', name)
    # 去掉剩余括号内容
    name = re.sub(r'[（(][^）)]*[）)]', '', name)
    for v in vol_keep:
        name += v
    # 去掉 _副本(数字) 等后缀
    name = re.sub(r'_\d*(副本|Copy|copy)', '', name)
    name = re.sub(r'[（(]\d*[）)]?_副本', '', name)
    name = re.sub(r'_[a-z0-9]+$', '', name)
    name = re.sub(r'^\d+_', '', name)
    # 去空格和书名号
    name = re.sub(r'[《》\s：:]', '', name)
    return name

def _extract_vol(name):
    """提取卷号/上下等信息并转为阿拉伯数字用于精确匹配"""
    import re
    cn_map = {'一':'1','二':'2','两':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','十':'10'}
    m = re.search(r'[（(]第?([一二两三四五六七八九十\d]+)[卷册部][）)]', name)
    if m:
        v = m.group(1)
        return cn_map.get(v, v)
    # 上下中 + 可能的后缀，如（上卷）（下册）（中）
    m = re.search(r'[（(]([上下中])[^）)]*[）)]', name)
    if m:
        return m.group(1)
    # 裸数字如（3）
    m = re.search(r'[（(](\d+)[）)]', name)
    if m:
        return m.group(1)
    # 无括号文件名中的卷号，如 日记卷  1_  或 远征纪实卷3
    m = re.search(r'[卷部册]\s*(\d+)', name)
    if m:
        return m.group(1)
    return ''

def _match_pdf(diary_name):
    """在 pdf 目录中查找与 diary_name 匹配的 PDF 文件"""
    pdf_dir = os.path.join(BASE_DIR, "抗战日记pdf版")
    if not os.path.isdir(pdf_dir):
        return None
    import re
    target = _clean_name(diary_name)
    target_vol = _extract_vol(diary_name)
    candidates = []
    for f in os.listdir(pdf_dir):
        if not f.lower().endswith('.pdf'):
            continue
        pdf_clean = _clean_name(f[:-4])
        pdf_vol = _extract_vol(f)
        # 基础字符重叠率
        common = sum(1 for ch in target if ch in pdf_clean)
        score = common / max(len(target), len(pdf_clean), 1)
        if target in pdf_clean or pdf_clean in target:
            score += 0.3
        # 卷号精确匹配加分（防止上下册/不同卷误配）
        if target_vol and pdf_vol and target_vol == pdf_vol:
            score += 0.5
        elif target_vol or pdf_vol:
            score -= 0.3  # 一方有卷号另一方没有，降权
        candidates.append((score, f, pdf_clean, pdf_vol))
    candidates.sort(key=lambda x: -x[0])
    if candidates and candidates[0][0] > 0.6:
        return os.path.join(pdf_dir, candidates[0][1])
    return None

def _get_page_offset(diary_name):
    """部分日记开头有封面/二维码，跳过这些页"""
    offsets = {
        "烽火晋察冀刘荣抗战日记": 2,
        "王恩茂日记——红军长征到": 2,
        "红军长征日记》陈伯钧": 2,
        "红军长征纪实丛书_日记卷（2）": 2,
    }
    if diary_name:
        for k, v in offsets.items():
            if k in diary_name:
                return v
    return 0

@st.cache_data
def get_diary_pdf_page_count(diary_name):
    """获取PDF总页数（扣除开头偏移）"""
    pdf_path = _match_pdf(diary_name)
    if not pdf_path:
        return 0
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        count = max(0, doc.page_count - _get_page_offset(diary_name))
        doc.close()
        return count
    except Exception:
        return 0

@st.cache_data
def get_diary_pdf_page_image(diary_name, page_num):
    """获取PDF页面图片（统一dpi渲染+裁剪黑边+方向校正）"""
    pdf_path = _match_pdf(diary_name)
    if not pdf_path:
        return None
    try:
        import pymupdf
        from PIL import Image, ImageOps
        import io

        offset = _get_page_offset(diary_name)
        doc = pymupdf.open(pdf_path)
        if page_num < 0 or page_num + offset >= doc.page_count:
            doc.close()
            return None
        page = doc[page_num + offset]

        # 统一渲染：林伯渠页面偏小用 dpi=300，其余用 dpi=200
        is_small_page = diary_name is not None and "林伯渠" in diary_name
        pix = page.get_pixmap(dpi=300 if is_small_page else 200)
        doc.close()
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        # 1) 颜色反转：暗图→白底黑字+对比度增强
        w, h = img.size
        dark_pixels = 0
        total = 0
        for y in range(h // 4, h * 3 // 4, max(1, h // 20)):
            for x in range(w // 4, w * 3 // 4, max(1, w // 20)):
                if sum(img.getpixel((x, y))[:3]) / 3 < 100:
                    dark_pixels += 1
                total += 1
        if dark_pixels / total > 0.5:
            img = ImageOps.invert(img)
            img = ImageOps.autocontrast(img, cutoff=2)

        # 2) 裁剪四周黑边
        img = _crop_borders(img)

        # 3) 手动旋转修正：个别日记PDF页面方向与实际内容方向不一致
        _NEED_ROTATION = {
            "《阵中日记》（东北人民解放军）",
        }
        if diary_name is not None and any(k in diary_name for k in _NEED_ROTATION):
            if img.size[1] > img.size[0]:
                img = img.rotate(90, expand=True, fillcolor="white")

        # 4) 底部水印裁剪：部分日记每页末尾有水印
        _WATERMARK_DIARIES = {
            "《烽火晋察冀刘荣抗战日记》":       70,
            "《王恩茂日记——南征北战》":         70,
            "《王恩茂日记——解放战争》":         70,
            "《王恩茂日记——抗日战争下》":       70,
            "《红军长征日记》陈伯钧、童小鹏、伍云甫": 80,
            "《红军长征纪实丛书_日记卷（2）》":  60,
        }
        if diary_name is not None:
            # 含中文引号的独立处理
            if "王恩茂日记" in diary_name and "红军长征" in diary_name:
                h0 = img.size[1]
                img = img.crop((0, 0, img.size[0], max(0, h0 - 70)))
            else:
                for k, crop_px in _WATERMARK_DIARIES.items():
                    if k in diary_name:
                        h0 = img.size[1]
                        img = img.crop((0, 0, img.size[0], max(0, h0 - crop_px)))
                        break

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _crop_borders(img):
    """裁剪四周黑边和白边，基于像素值范围检测真实内容边界"""
    w, h = img.size
    step = max(1, min(w, h) // 100)

    def _scan_y(y):
        vals = [sum(img.getpixel((x, y))[:3]) for x in range(0, w, step)]
        rng = max(vals) - min(vals)
        if rng > 25:
            return rng  # 有明显内容
        avg = sum(vals) / len(vals) / 3
        return 0 if avg < 30 or avg > 252 else rng

    def _scan_x(x):
        vals = [sum(img.getpixel((x, y))[:3]) for y in range(0, h, step)]
        rng = max(vals) - min(vals)
        if rng > 25:
            return rng
        avg = sum(vals) / len(vals) / 3
        return 0 if avg < 30 or avg > 252 else rng

    # 上
    top = 0
    for y in range(0, h, step):
        if _scan_y(y) > 25:
            top = max(0, y - step)
            break
        top = y + step

    # 下
    bottom = h - 1
    for y in range(h - 1, -1, -step):
        if _scan_y(y) > 25:
            bottom = min(h - 1, y + step)
            break
        bottom = y - step

    # 左
    left = 0
    for x in range(0, w, step):
        if _scan_x(x) > 25:
            left = max(0, x - step)
            break
        left = x + step

    # 右
    right = w - 1
    for x in range(w - 1, -1, -step):
        if _scan_x(x) > 25:
            right = min(w - 1, x + step)
            break
        right = x - step

    pad = 2
    cropped = img.crop((max(0, left - pad), max(0, top - pad),
                        min(w, right + 1 + pad), min(h, bottom + 1 + pad)))

    # 防止裁剪过小（比如章节过渡页只有一行文字）：至少保留原图40%尺寸
    cw, ch = cropped.size
    if cw < w * 0.25 or ch < h * 0.25:
        # 退回宽松裁剪：取原图中心80%区域
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        cropped = img.crop((margin_x, margin_y, w - margin_x, h - margin_y))

    return cropped

@st.cache_data
def get_diary_raw_text(diary_name):
    """读取日记全文（TXT回退方案）"""
    import re
    diary_dir = os.path.join(BASE_DIR, "抗战日记（43本清洗版）")
    target = re.sub(r'\s+', '', diary_name)
    for f in os.listdir(diary_dir):
        if not f.endswith('.txt'):
            continue
        cleaned = re.sub(r'\s+', '', f[:-4])
        if cleaned == target:
            filepath = os.path.join(diary_dir, f)
            with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                return fh.read()
    return None

def get_diary_entries(diary_name):
    """获取某本日记的全部条目，按时间排序"""
    import pickle as _pickle
    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = _pickle.load(f)
    entries = [e for e in meta if e.get('diary_name') == diary_name]
    entries.sort(key=lambda e: (e.get('year') or 9999, e.get('month') or 99, e.get('day') or 99))
    return entries


@st.cache_data
def browse_all_entries(min_year=None, max_year=None, diary_names=None):
    """浏览全部条目，按时间排序，可选按年代/日记名筛选"""
    import pickle as _pickle
    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = _pickle.load(f)
    result = []
    for e in meta:
        if min_year and (e.get('year') or 9999) < min_year: continue
        if max_year and (e.get('year') or 0) > max_year: continue
        if diary_names and e.get('diary_name') not in diary_names: continue
        result.append(e)
    result.sort(key=lambda e: (e.get('year') or 9999, e.get('month') or 99, e.get('day') or 99))
    return result


@st.cache_data
def load_sentiment_extremes(n=5):
    """加载情感极端案例（最正向/负向条目）"""
    import pickle as _pickle
    import json
    SENT_FILE = os.path.join(BASE_DIR, "parsed_data/sentiment_results.json")
    if not os.path.exists(SENT_FILE):
        return None
    with open(SENT_FILE) as f:
        sent_data = json.load(f)
    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'rb') as f:
        meta = _pickle.load(f)
    entries = sent_data.get("entries", [])
    sorted_entries = sorted(entries, key=lambda x: x['score'])
    result = {"positive": [], "negative": []}
    for es in sorted_entries[:n]:
        idx = es['idx']
        e = meta[idx] if idx < len(meta) else {}
        result["negative"].append({
            "score": es['score'], "diary_name": e.get('diary_name',''),
            "year": e.get('year'), "month": e.get('month'), "day": e.get('day'),
            "date_raw": e.get('date_raw',''), "text": (e.get('text','') or '')[:300],
            "neg_words": es.get('neg_words', [])[:10],
        })
    for es in sorted_entries[-n:][::-1]:
        idx = es['idx']
        e = meta[idx] if idx < len(meta) else {}
        result["positive"].append({
            "score": es['score'], "diary_name": e.get('diary_name',''),
            "year": e.get('year'), "month": e.get('month'), "day": e.get('day'),
            "date_raw": e.get('date_raw',''), "text": (e.get('text','') or '')[:300],
            "pos_words": es.get('pos_words', [])[:10],
        })
    return result


def _fmt_date(e):
    """格式化日期"""
    if e.get('year') and e.get('month') and e.get('day'):
        return f"{e['year']}年{e['month']}月{e['day']}日"
    return e.get('date_raw') or ''


def _export_csv(results, query, filter_info, include_rel=False):
    """导出结果为 CSV（可选包含关联条目）"""
    import csv, io
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(['序号', '类型', '日记名', '日期', '年份', '分类', '子标签', '天气', '匹配度', '正文'])
    for i, r in enumerate(results, 1):
        w.writerow([
            i, '正文',
            r['diary_name'], _fmt_date(r), r.get('year', ''),
            r.get('category', ''), r.get('sub_tag', ''),
            r.get('weather', ''), f"{r.get('score', 1)*100:.0f}%", r['text'],
        ])
        if include_rel:
            rel = get_related(r['diary_name'], r.get('year'), r.get('month'), r.get('day'),
                              (r.get('text') or '')[:50], r.get('category'), r.get('sub_tag'))
            for _, re in rel:
                w.writerow([
                    i, f'关联',
                    re['diary_name'], _fmt_date(re), re.get('year', ''),
                    re.get('category', ''), re.get('sub_tag', ''),
                    re.get('weather', ''), '', re['text'],
                ])
    return out.getvalue()


def _export_txt(results, query, filter_info, include_rel=False):
    """导出结果为 TXT（可选包含关联条目）"""
    lines = []
    lines.append(f"革命日记检索 - 搜索结果")
    lines.append(f"{'搜索词: ' + query if query else '浏览模式'}")
    if filter_info:
        lines.append(f"筛选: {filter_info}")
    lines.append(f"共 {len(results)} 条结果")
    lines.append("=" * 70)
    for i, r in enumerate(results, 1):
        ds = _fmt_date(r)
        wt = f" 天气:{r['weather']}" if r.get('weather') else ""
        cat = ""
        if r.get('category'):
            cat = f" [{r['category']}"
            if r.get('sub_tag'):
                cat += f" / {r['sub_tag']}"
            cat += "]"
        lines.append(f"\n[{i}] 《{r['diary_name']}》 {ds}{wt}{cat}  ({r.get('score', 1)*100:.0f}%)")
        lines.append("-" * 50)
        lines.append(r['text'])
        if include_rel:
            rel = get_related(r['diary_name'], r.get('year'), r.get('month'), r.get('day'),
                              (r.get('text') or '')[:50], r.get('category'), r.get('sub_tag'))
            if rel:
                lines.append(f"\n  ── 关联条目 ──")
                for _, re in rel:
                    rds = _fmt_date(re)
                    rcat = f"[{re.get('category','')}" + (f" / {re['sub_tag']}]" if re.get('sub_tag') else "]")
                    lines.append(f"  《{re['diary_name']}》 {rds} {rcat}")
                    lines.append(f"  {re['text'][:200]}{'…' if len(re['text']) > 200 else ''}")
    lines.append("\n" + "=" * 70)
    return '\n'.join(lines)


@st.cache_data
def get_related(diary_name, year, month, day, text_head, category, sub_tag, top_k=6):
    """查找关联条目：同分类 + 同子标签 / 同年 / 同日记本"""
    import pickle as _pickle
    meta_path = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = _pickle.load(f)
    scored = []
    for e in meta:
        # 排除自身
        if (e.get('diary_name') == diary_name and e.get('year') == year
                and e.get('month') == month and e.get('day') == day
                and (e.get('text') or '')[:50] == text_head):
            continue
        score = 0
        ec, ey = e.get('category', ''), e.get('year')
        if category and ec == category:
            score += 3
            if sub_tag and e.get('sub_tag') == sub_tag:
                score += 4
        if year and ey:
            if ey == year:
                score += 2
            elif abs(ey - year) <= 2:
                score += 1
        if e.get('diary_name') == diary_name:
            score += 1
        if score >= 3:
            scored.append((score, e))
    scored.sort(key=lambda x: (-x[0], -(x[1].get('year') or 0), -(x[1].get('month') or 0)))
    return scored[:top_k]


def _call_deepseek_summary(entries, api_key, max_entries=20):
    """调用 DeepSeek API 为日记条目生成摘要"""
    import requests, json
    if not api_key:
        return None

    # 限制条目数，避免超长上下文
    if len(entries) > max_entries:
        entries = entries[:max_entries]

    texts = []
    for i, e in enumerate(entries, 1):
        ds = _fmt_date(e)
        cat = f" [{e.get('category','')}" + (f" / {e['sub_tag']}]" if e.get('sub_tag') else "]") if e.get('category') else ""
        text = e['text']
        if len(text) > 800:
            text = text[:800] + "…"
        texts.append(f"[{i}] 《{e['diary_name']}》 {ds}{cat}\n{text}")

    prompt = (
        f"以下是革命日记中的 {len(entries)} 条记录。请用一段话（约200-400字）概括主要事件、"
        f"主题内容和情感倾向，帮助读者快速了解日记内容。\n\n"
        + "\n\n---\n\n".join(texts)
    )

    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是历史研究助手，擅长概括革命日记内容。回答简洁、准确、有条理。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


st.title("📖 革命日记全文检索")
st.markdown("jieba 分词 + TF-IDF + DeepSeek 查询扩展 | 六大类+子标签分类筛选")

# 侧栏
with st.sidebar:
    mode = st.radio("模式", ["🔍 检索", "📊 统计看板", "🗺️ 地理分布", "📚 全文浏览", "💬 情感分析", "🗺️ 事件脉络"], horizontal=True, label_visibility="collapsed")

    if mode == "📊 统计看板":
        st.markdown("### 📊 统计看板")
        st.caption("基于 23,834 条结构化日记数据的统计分析")
        st.divider()
        col1, col2, col3 = st.columns(3)
        df = load_stats_data()
        if df is not None:
            col1.metric("总条目", f"{len(df):,}")
            col2.metric("日记本", df['diary_name'].nunique())
            covered = (df['category'] != '未分类').mean()
            col3.metric("分类覆盖率", f"{covered*100:.1f}%")
            st.divider()
            st.caption("分类概览")
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts = cat_counts[~cat_counts['category'].isin(['未分类', ''])]
            cat_counts.columns = ['category', 'count']
            st.dataframe(cat_counts, use_container_width=True, hide_index=True)
    elif mode == "🗺️ 地理分布":
        st.markdown("### 🗺️ 地理分布")
        st.caption("基于地点信息的空间分布分析")
        st.divider()
        df = load_stats_data()
        if df is not None:
            diary_list_map = sorted(df['diary_name'].unique())
            map_diary = st.selectbox("日记本", ["全部日记"] + diary_list_map, index=0, key="map_diary_standalone")
            diary_arg = map_diary if map_diary != "全部日记" else None
            if diary_arg:
                all_tags_for_map = get_tags_with_locations(diary_arg)
                tag_options = ["全部"] + all_tags_for_map
            else:
                all_tags_for_map = get_clean_sub_tags(df)
                tag_options = ["全部"] + all_tags_for_map
            st.selectbox("子标签", tag_options, index=0, key="map_tag_standalone")
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                st.selectbox("起始年", [None] + sorted(df['year'].dropna().unique().astype(int)), index=0, key="map_min_y_standalone")
            with col_y2:
                st.selectbox("截止年", [None] + sorted(df['year'].dropna().unique().astype(int)), index=0, key="map_max_y_standalone")

        # 在侧栏预计算行程
        s_diary = st.session_state.get("map_diary_standalone", "全部日记")
        s_tag = st.session_state.get("map_tag_standalone", "全部")
        s_diary_arg = s_diary if s_diary != "全部日记" else None
        s_tag_arg = s_tag if s_tag and s_tag != "全部" else None
        if s_diary_arg and s_tag_arg:
            s_route = extract_route_data(s_diary_arg, s_tag_arg)
            if s_route:
                s_deduped = [s_route[0]]
                for p in s_route[1:]:
                    if p["location"] != s_deduped[-1]["location"]:
                        s_deduped.append(p)
                s_route = s_deduped
                if len(s_route) >= 2:
                    st.session_state["geo_trips"] = split_route_into_trips(s_route, diary_name=s_diary_arg)
                    s_fk = f"{s_diary_arg}|{s_tag_arg}"
                    if st.session_state.get("geo_filter_key") != s_fk:
                        st.session_state["geo_filter_key"] = s_fk
                        st.session_state["geo_trip_idx"] = -1
                else:
                    st.session_state["geo_trips"] = None
            else:
                st.session_state["geo_trips"] = None
        else:
            st.session_state["geo_trips"] = None

        # 行程导航 UI
        s_trips = st.session_state.get("geo_trips", None)
        if s_trips and len(s_trips) >= 1:
            st.divider()
            st.markdown("**⭐ 行程导航**")
            opts = ["总览（全部）"]
            for i, t in enumerate(s_trips):
                opts.append(f"行程 {i+1}: {t[0]['date']} ~ {t[-1]['date']}（{len(t)}站）")
            s_trip_idx = st.session_state.get("geo_trip_idx", -1)
            s_sel = st.selectbox("查看", opts, index=s_trip_idx + 1,
                               label_visibility="collapsed", key="geo_trip_sel")
            s_new_idx = opts.index(s_sel) - 1
            if s_new_idx != s_trip_idx:
                st.session_state["geo_trip_idx"] = s_new_idx
                st.rerun()

    elif mode == "📚 全文浏览":
        st.markdown("### 📚 全文浏览")
        diary_list = get_diary_list()
        scope = st.selectbox("浏览范围", ["全部日记"] + diary_list, key="browse_scope")
        st.divider()
        if scope == "全部日记":
            st.caption("按年代+日记名浏览全部内容")
            diaries_list = diary_list
            browse_diaries = st.multiselect("筛选日记（不选则全部）", diaries_list, default=[], key="browse_diaries")
            diaries, years = get_meta()
            col1, col2 = st.columns(2)
            with col1:
                browse_min_y = st.selectbox("起始年", [None] + years, index=0, key="browse_min_y")
            with col2:
                browse_max_y = st.selectbox("截止年", [None] + years, index=0, key="browse_max_y")
        else:
            scoped_entries = get_diary_entries(scope)
            st.metric("条目数", len(scoped_entries))
            if scoped_entries:
                y1 = scoped_entries[0].get('year', '?')
                y2 = scoped_entries[-1].get('year', '?')
                st.caption(f"时间跨度: {y1} - {y2}")
            browse_read_mode = st.radio("阅读模式", ["分类浏览", "全文阅读"], horizontal=True, label_visibility="collapsed", key="browse_read_mode")
            st.divider()

            st.markdown("**分类筛选**")
            cats = get_categories()
            browse_cat = st.selectbox("大类", [None] + cats, index=0, key="browse_cat")
            sub_tags = []
            if browse_cat:
                engine = get_engine()
                if engine:
                    sub_tags = engine.get_sub_tags(browse_cat)
            browse_tag = st.selectbox("子标签", [None] + sub_tags, index=0, key="browse_tag")
            st.divider()
            st.markdown("**📊 单本日记统计**")
            df_s = load_stats_data()
            if df_s is not None:
                diary_df = df_s[df_s['diary_name'] == scope]
                if not diary_df.empty:
                    import altair as alt
                    # 分类分布
                    cat_counts = diary_df['category'].value_counts().reset_index()
                    cat_counts.columns = ['category', 'count']
                    cat_counts = cat_counts[~cat_counts['category'].isin(['未分类', ''])]
                    if not cat_counts.empty:
                        cat_chart = alt.Chart(cat_counts).mark_bar().encode(
                            x=alt.X('count:Q', title=None),
                            y=alt.Y('category:N', title=None, sort='-x'),
                            color=alt.Color('category:N', legend=None),
                            tooltip=['category', 'count'],
                        ).properties(height=120, title="分类分布")
                        st.altair_chart(cat_chart, use_container_width=True)
                    # 年代分布
                    yearly = diary_df.groupby('year').size().reset_index(name='count')
                    yearly = yearly[yearly['year'].notna()]
                    yearly['year'] = yearly['year'].astype(int)
                    if not yearly.empty:
                        year_chart = alt.Chart(yearly).mark_bar().encode(
                            x=alt.X('year:O', title=None),
                            y=alt.Y('count:Q', title=None),
                            tooltip=['year', 'count'],
                        ).properties(height=120, title="年代分布")
                        st.altair_chart(year_chart, use_container_width=True)
                    # 分类×年代堆叠
                    cat_year = diary_df.groupby(['year', 'category']).size().reset_index(name='count')
                    cat_year = cat_year[cat_year['year'].notna() & ~cat_year['category'].isin(['未分类', ''])]
                    if not cat_year.empty:
                        cat_year['year'] = cat_year['year'].astype(int)
                        pivot = cat_year.pivot(index='year', columns='category', values='count').fillna(0).astype(int)
                        st.caption("分类 × 年代分布")
                        st.bar_chart(pivot, height=150)
        browse_page_size = st.selectbox("每页条数", [10, 20, 50, 100], index=0, key="browse_page_size")
        st.divider()
        st.markdown("**🤖 AI 摘要**")
        st.toggle("启用DeepSeek摘要", value=False, key="browse_use_summary",
                  help="用 DeepSeek 为所选日记生成内容提要")
        if st.session_state.get("browse_use_summary", False):
            st.text_input("API Key", type="password", key="browse_api_key",
                          help="从 platform.deepseek.com 获取")
            st.caption("支持单篇或批量选中后生成摘要")
    elif mode == "💬 情感分析":
        st.markdown("### 💬 情感分析")
        st.caption("v2 三级强度词典 + 否定词处理 + 情感子类分析")
        st.divider()
        SENTIMENT_FILE = os.path.join(BASE_DIR, "parsed_data/sentiment_results.json")
        if os.path.exists(SENTIMENT_FILE):
            import json as _json
            with open(SENTIMENT_FILE) as f:
                sa_data = _json.load(f)
            st.metric("总条目", f"{sa_data['total']:,}")
            col1, col2, col3 = st.columns(3)
            col1.metric("😊 正向", f"{sa_data['stats']['positive_pct']}%")
            col2.metric("😐 中立", f"{sa_data['stats']['neutral_pct']}%")
            col3.metric("😟 负向", f"{sa_data['stats']['negative_pct']}%")
            _is_v2 = sa_data.get('method', '').startswith('enhanced')
            _intensity_str = f" 强度:{sa_data['stats'].get('mean_intensity', '—')}"
            st.caption(f"词典: 正{sa_data['lexicon_size']['positive']}词 / 负{sa_data['lexicon_size']['negative']}词{' (v2增强版)' if _is_v2 else ''}{_intensity_str}")
            st.divider()
            sa_view = st.radio("查看方式", ["情感演化", "总体分布", "按年份趋势", "按分类对比", "按日记本对比", "极端案例", "情感画像"],
                              horizontal=True, key="sa_view")
            st.divider()
            # 筛选：仅对特定视图显示
            show_cat_filter = st.session_state.get("sa_view", "情感演化") in ("总体分布", "按分类对比", "按日记本对比", "极端案例")
            show_year_filter = st.session_state.get("sa_view", "情感演化") in ("总体分布", "按年份趋势", "按分类对比", "按日记本对比", "极端案例")
            if show_cat_filter or show_year_filter:
                st.markdown("**筛选**")
                if show_cat_filter:
                    sa_cats = list(sa_data.get('by_category', {}).keys())
                    sa_cats = [c for c in sa_cats if c not in ('未分类', '')]
                    st.selectbox("分类", [None] + sa_cats, index=0, key="sa_cat")
                if show_year_filter:
                    sa_years = sorted([int(y) for y in sa_data.get('by_year', {}).keys()])
                    col_y1, col_y2 = st.columns(2)
                    with col_y1:
                        st.selectbox("起始年", [None] + sa_years, index=0, key="sa_min_y")
                    with col_y2:
                        st.selectbox("截止年", [None] + sa_years, index=0, key="sa_max_y")
            # 情感演化专用：日记选择+参数
            if st.session_state.get("sa_view", "情感演化") == "情感演化":
                st.markdown("**日记选择**")
                d_list = get_diary_list()
                st.selectbox("日记本", d_list, index=0, key="sa_diary")
                st.slider("平滑窗口（天）", 7, 365, 90, key="sa_evo_window")
                st.checkbox("标注国家级历史事件", value=True, key="sa_natl_events")
        else:
            st.warning("请先运行 sentiment_analysis.py")
    elif mode == "🗺️ 事件脉络":
        st.markdown("### 🗺️ 事件脉络")
        ev_sub_mode = st.radio("子功能", ["事件脉络", "知识图谱"], horizontal=True, key="ev_sub_mode", label_visibility="collapsed")

        if ev_sub_mode == "事件脉络":
            st.caption("跨日记事件详情")
            st.divider()
            EVENT_FILE = os.path.join(BASE_DIR, "parsed_data/event_timeline.json")
            if os.path.exists(EVENT_FILE):
                ev_data = load_event_data(500, os.path.getmtime(EVENT_FILE))
                if ev_data:
                    stats = ev_data["statistics"]
                    st.metric("事件簇", f"{stats['total_event_clusters']:,}")
                    st.metric("跨日记事件", stats['cross_diary_event_count'], help=f"占比 {stats['cross_diary_event_pct']}%")
                    st.divider()
                    st.selectbox(
                        "事件门类",
                        ["全部", "军事作战", "组织建设", "群众运动", "政权建设", "文化建设", "日常生活", "其他"],
                        index=0, key="ev_category",
                    )
                    st.slider("最低重要性", 0.0, 1.0, 0.3, 0.05, key="ev_min_imp")
        else:
            st.caption("子标签共现知识图谱")
            st.divider()
            cl_data = load_classified(os.path.getmtime(CLASSIFIED_FILE))
            if cl_data:
                sub_tags = sorted(set(e["sub_tag"] for e in cl_data["entries"] if e.get("sub_tag") and e["sub_tag"] not in ("其他", "", "未分类") and "人事" not in e["sub_tag"]))
                sel_sub = st.selectbox("子标签", sub_tags, key="wc_subtag")
                if st.button("🔄 生成知识图谱", type="primary", use_container_width=True):
                    with st.spinner(f"分析「{sel_sub}」共现词汇..."):
                        wf = compute_word_freq(sel_sub, os.path.getmtime(CLASSIFIED_FILE))
                        st.session_state["wc_data"] = (sel_sub, wf)
    else:
        # 🔍 检索 模式 (以及 💬 情感分析)
        st.header("⚙️ 设置")
        use_llm = st.toggle("DeepSeek 语义扩展", value=False,
                           help="开启后用 DeepSeek 扩展查询同义词，语义更准")
        api_key = st.text_input("API Key（开启后需填入）", type="password",
                               help="从 platform.deepseek.com 获取")

        st.divider()
        st.header("🔍 筛选")

        # 分类筛选
        cats = get_categories()
        sel_cat = st.selectbox("大类", [None] + cats, index=0, key="cat_select")

        sub_tags = []
        if sel_cat and os.path.exists(CLASSIFIED_FILE):
            engine = get_engine()
            if engine:
                sub_tags = engine.get_sub_tags(sel_cat)
        sel_tag = st.selectbox("子标签", [None] + sub_tags, index=0, key="tag_select")

        diaries, years = get_meta()
        sel_diaries = st.multiselect("日记", diaries, default=[])
        col1, col2 = st.columns(2)
        with col1:
            min_y = st.selectbox("起始年", [None] + years, index=0)
        with col2:
            max_y = st.selectbox("截止年", [None] + years, index=0)

        page_size = st.selectbox("每页条数", [10, 20, 50, 100], index=1)

        if st.button("🔄 重建索引"):
            with st.spinner("..."):
                build_index()
            st.success("完成")
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"📊 {load_stats_data().__len__() if load_stats_data() is not None else '?'} 条 | 本地运行")


# =========================================================================
# 统计看板
# =========================================================================
if mode == "📊 统计看板":
    import altair as alt
    import pandas as pd

    df = load_stats_data()
    if df is None:
        st.warning("请先构建索引")
        st.stop()

    # --- 分类分布 ---
    st.subheader("📂 分类分布")
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['category', 'count']
    cat_counts = cat_counts[~cat_counts['category'].isin(['未分类', ''])]
    cat_counts = cat_counts.sort_values('count', ascending=True)
    bars = alt.Chart(cat_counts).mark_bar().encode(
        x=alt.X('count:Q', title='条目数'),
        y=alt.Y('category:N', title=None, sort='-x'),
        color=alt.Color('category:N', legend=None,
                        scale=alt.Scale(domain=['日常生活', '军事作战', '组织建设', '群众运动', '政权建设', '文化建设'],
                                        range=['#4a90d9', '#e74c3c', '#e8783a', '#50b86c', '#9b59b6', '#e67e22'])),
        tooltip=['category', 'count'],
    ).properties(height=200)
    st.altair_chart(bars, use_container_width=True)

    # --- 年代分布 ---
    st.subheader("📅 年代分布")
    yearly = df.groupby('year').size().reset_index(name='count')
    yearly = yearly[yearly['year'].notna()]
    yearly['year'] = yearly['year'].astype(int)
    year_line = alt.Chart(yearly).mark_line(point=True).encode(
        x=alt.X('year:O', title='年份'),
        y=alt.Y('count:Q', title='条目数'),
        tooltip=['year', 'count'],
    ).properties(height=250)
    st.altair_chart(year_line, use_container_width=True)

    # --- 分类 × 年代堆叠图 ---
    st.subheader("📊 分类 × 年代分布")
    cat_year = df.groupby(['year', 'category']).size().reset_index(name='count')
    cat_year = cat_year[cat_year['year'].notna() & ~cat_year['category'].isin(['未分类', ''])]
    cat_year['year'] = cat_year['year'].astype(int)
    pivot = cat_year.pivot(index='year', columns='category', values='count').fillna(0).astype(int)
    st.bar_chart(pivot, height=350)

    # --- 子标签年度分布 ---
    st.subheader("🏷️ 子标签年度分布")
    all_tags = get_clean_sub_tags(df)
    sel_tag = st.selectbox("选择子标签", all_tags, index=all_tags.index("住宿") if "住宿" in all_tags else 0,
                          key="stat_tag")
    tag_yearly = df[df['sub_tag'].apply(lambda t: match_sub_tag(t, sel_tag))].groupby('year').size().reset_index(name='count')
    tag_yearly = tag_yearly[tag_yearly['year'].notna()]
    tag_yearly['year'] = tag_yearly['year'].astype(int)
    if not tag_yearly.empty:
        tag_line = alt.Chart(tag_yearly).mark_line(point=True, color='#e45756').encode(
            x=alt.X('year:O', title='年份'),
            y=alt.Y('count:Q', title='提及次数'),
            tooltip=['year', 'count'],
        ).properties(height=250)
        st.altair_chart(tag_line, use_container_width=True)
        peak = tag_yearly.loc[tag_yearly['count'].idxmax()]
        st.caption(f"📈 峰值: **{peak['year']}** 年（{peak['count']} 次） | "
                   f"年均: **{tag_yearly['count'].mean():.1f}** 次")
    else:
        st.info("该子标签暂无年度数据")

    st.caption(f"数据来源: parsed_data/diaries_structured.json + classified_entries.json | 共 {len(df):,} 条")
    st.stop()


# =========================================================================
# 地理热力图（独立模式）
# =========================================================================
elif mode == "🗺️ 地理分布":
    import pandas as pd

    df = load_stats_data()
    if df is None:
        st.warning("请先构建索引")
        st.stop()

    map_diary = st.session_state.get("map_diary_standalone", "全部日记")
    map_tag = st.session_state.get("map_tag_standalone", "全部")
    map_min_y = st.session_state.get("map_min_y_standalone")
    map_max_y = st.session_state.get("map_max_y_standalone")

    diary_arg = map_diary if map_diary != "全部日记" else None
    sub_tag_arg = map_tag if map_tag and map_tag != "全部" else None
    year_range_arg = (map_min_y, map_max_y) if map_min_y or map_max_y else None

    if not (diary_arg and sub_tag_arg):
        st.session_state["geo_trips"] = None

    st.subheader("🗺️ 地理热力图")

    # 单本日记 + 有子标签 → 尝试显示路线，按行程分段
    route_shown = False
    if diary_arg and sub_tag_arg:
        route_data = extract_route_data(diary_arg, sub_tag_arg)
        if route_data:
            # 去重连续同地点
            deduped = [route_data[0]]
            for p in route_data[1:]:
                if p["location"] != deduped[-1]["location"]:
                    deduped.append(p)
            route_data = deduped

            if len(route_data) >= 2:
                trips = st.session_state.get("geo_trips", None)
                trip_idx = st.session_state.get("geo_trip_idx", -1)
                display = route_data
                title_suffix = ""
                if trip_idx >= 0 and trips and trip_idx < len(trips):
                    display = trips[trip_idx]
                    title_suffix = f"（行程 {trip_idx+1}/{len(trips)}）"

                route_shown = True
                st.markdown(f"**🗺️ 「{sub_tag_arg}」路线图**{title_suffix}")
                leaflet_html = _build_route_map(display)
                components.html(leaflet_html, height=500)
                route_desc = " → ".join(p["location"] for p in display)
                date_range = f"{display[0]['date']} ～ {display[-1]['date']}"
                st.markdown(f"**路线：** {route_desc}")
                st.caption(f"📅 {date_range} ｜ 共 {len(display)} 站，{len(display)-1} 段行程")
            else:
                st.session_state["geo_trips"] = None
                st.info("该路线只有单一地点")
        else:
            st.session_state["geo_trips"] = None

    if not route_shown:
        loc_data = extract_diary_locations(sub_tag=sub_tag_arg, year_range=year_range_arg, diary_name=diary_arg)
        if loc_data:
            point_html = _build_point_map(loc_data)
            components.html(point_html, height=500)
            top_locs = sorted(loc_data, key=lambda x: -x['count'])[:15]
            loc_table = pd.DataFrame([{k: v for k, v in p.items() if k in ('location', 'lat', 'lon', 'count')} for p in top_locs])
            loc_table.columns = ['地点', '纬度', '经度', '提及次数']
            st.caption(f"共 {len(loc_data)} 个地点 | Top 15")
            st.dataframe(loc_table[['地点', '提及次数']], use_container_width=True, hide_index=True)
        else:
            st.info("所选条件下暂无地点数据")
    st.stop()


# =========================================================================
# 全文浏览
# =========================================================================
elif mode == "📚 全文浏览":
    scope = st.session_state.get("browse_scope", "全部日记")

    if scope == "全部日记":
        browse_min_y = st.session_state.get("browse_min_y")
        browse_max_y = st.session_state.get("browse_max_y")
        browse_diaries = st.session_state.get("browse_diaries", [])
        entries = browse_all_entries(
            min_year=browse_min_y,
            max_year=browse_max_y,
            diary_names=browse_diaries if browse_diaries else None,
        )
        if not entries:
            st.info("没有符合条件的条目，请调整筛选条件")
            st.stop()
        filter_parts_b = []
        if browse_min_y: filter_parts_b.append(f"≥{browse_min_y}")
        if browse_max_y: filter_parts_b.append(f"≤{browse_max_y}")
        if browse_diaries: filter_parts_b.append(f"{len(browse_diaries)}本日记")
        filter_info_b = " | ".join(filter_parts_b) if filter_parts_b else "全部条目"
        st.subheader("📚 全文浏览")
        st.caption(f"{filter_info_b} | 共 {len(entries)} 条")
    else:
        browse_read_mode = st.session_state.get("browse_read_mode", "分类浏览")

        # 全文阅读模式：展示PDF页面图片（翻页阅读）
        if browse_read_mode == "全文阅读":
            pdf_total = get_diary_pdf_page_count(scope)
            if pdf_total > 0:
                page = st.session_state.get("ebook_page", 0)
                if page >= pdf_total:
                    page = 0
                st.subheader(f"📖 《{scope}》")
                img = get_diary_pdf_page_image(scope, page)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.error("页面渲染失败")
                col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                with col_p1:
                    if st.button("◀ 上一页", disabled=(page == 0), key="ebook_prev", use_container_width=True):
                        st.session_state["ebook_page"] = page - 1
                        st.rerun()
                with col_p2:
                    st.markdown(f"<div style='text-align:center;padding:8px'>第 {page+1}/{pdf_total} 页</div>", unsafe_allow_html=True)
                with col_p3:
                    if st.button("下一页 ▶", disabled=(page >= pdf_total - 1), key="ebook_next", use_container_width=True):
                        st.session_state["ebook_page"] = page + 1
                        st.rerun()
                # 页面跳转
                jc1, jc2, jc3, jc4 = st.columns([2, 1, 1, 2])
                with jc2:
                    jump_input = st.number_input("", min_value=1, max_value=pdf_total, value=page+1, label_visibility="collapsed", key="ebook_jump_input")
                with jc3:
                    if st.button("确认", use_container_width=True):
                        st.session_state["ebook_page"] = jump_input - 1
                        st.rerun()
            else:
                # 无PDF，回退到TXT文本阅读
                raw = get_diary_raw_text(scope)
                if raw is None:
                    st.warning("未找到原文文件")
                    st.stop()
                lines = raw.split('\n')
                body = '\n'.join(lines[1:]).strip() if lines else raw
                page_size = 3000
                total_chars = len(body)
                total_pages = max(1, (total_chars + page_size - 1) // page_size)
                page = st.session_state.get("ebook_page", 0)
                if page >= total_pages:
                    page = 0
                start = page * page_size
                end = min(start + page_size, total_chars)
                st.subheader(f"📖 《{scope}》（TXT文本）")
                st.text_area("", body[start:end], height=600, label_visibility="collapsed")
                col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                with col_p1:
                    if st.button("◀ 上一页", disabled=(page == 0), key="ebook_prev", use_container_width=True):
                        st.session_state["ebook_page"] = page - 1
                        st.rerun()
                with col_p2:
                    st.markdown(f"<div style='text-align:center;padding:8px'>第 {page+1}/{total_pages} 页（共 {total_chars:,} 字）</div>", unsafe_allow_html=True)
                with col_p3:
                    if st.button("下一页 ▶", disabled=(page >= total_pages - 1), key="ebook_next", use_container_width=True):
                        st.session_state["ebook_page"] = page + 1
                        st.rerun()
            st.stop()

        browse_cat = st.session_state.get("browse_cat")
        browse_tag = st.session_state.get("browse_tag")
        all_entries = get_diary_entries(scope)
        if not all_entries:
            st.warning("该日记暂无条目")
            st.stop()
        if browse_cat:
            if browse_tag:
                entries = [e for e in all_entries if e.get('category') == browse_cat and e.get('sub_tag') == browse_tag]
            else:
                entries = [e for e in all_entries if e.get('category') == browse_cat]
        else:
            entries = all_entries
        if not entries:
            st.warning("无匹配条目，请调整分类条件")
            st.stop()

        st.subheader(f"📖 《{scope}》")
        y1 = all_entries[0].get('year', '?')
        y2 = all_entries[-1].get('year', '?')
        extra = "" if len(entries) == len(all_entries) else f"（筛选后 {len(entries)} 条）"
        st.caption(f"共 {len(all_entries)} 条 | {y1} - {y2} {extra}")
        # 单本统计（分类分布小图）
        df_s = load_stats_data()
        if df_s is not None and not browse_cat:
            diary_df = df_s[df_s['diary_name'] == scope]
            if not diary_df.empty:
                cat_counts = diary_df['category'].value_counts()
                cat_counts = cat_counts[~cat_counts.index.isin(['未分类', ''])]
                if not cat_counts.empty:
                    st.bar_chart(cat_counts, height=150)
        filter_info_b = f"《{scope}》"
        if browse_cat:
            filter_info_b += f" | {browse_cat}"
            if browse_tag:
                filter_info_b += f" / {browse_tag}"

    total = len(entries)
    browse_page_size = st.session_state.get("browse_page_size", 10)
    total_pages = max(1, (total + browse_page_size - 1) // browse_page_size)
    page = st.session_state.get("browse_page", 0)
    if page >= total_pages:
        page = 0

    # 切换日记/筛选时重置到第一页
    browse_page_key = scope + str(st.session_state.get("browse_cat", "")) + str(st.session_state.get("browse_tag", ""))
    if st.session_state.get("last_browse_key") != browse_page_key:
        st.session_state["browse_page"] = 0
        st.session_state["last_browse_key"] = browse_page_key
        page = 0

    start = page * browse_page_size
    end = min(start + browse_page_size, total)

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    if col_prev.button("◀ 上一页", disabled=(page == 0), use_container_width=True):
        st.session_state["browse_page"] = max(0, page - 1)
        st.rerun()
    col_info.markdown(f"<div style='text-align:center;padding:8px;font-size:1.1em'>第 {page+1} / {total_pages} 页（共 {total} 条）</div>",
                      unsafe_allow_html=True)
    if col_next.button("下一页 ▶", disabled=(page >= total_pages - 1), use_container_width=True):
        st.session_state["browse_page"] = min(total_pages - 1, page + 1)
        st.rerun()

    st.divider()

    # --- 导出 ---
    if scope == "全部日记":
        browse_sel_key = "browse_sel"
    else:
        browse_sel_key = "browse_sel_" + scope

    if browse_sel_key not in st.session_state:
        st.session_state[browse_sel_key] = set()

    def toggle_browse_sel(bidx):
        s = st.session_state[browse_sel_key]
        if bidx in s:
            s.remove(bidx)
        else:
            s.add(bidx)

    re1, re2, re3, re4 = st.columns([1, 1, 1, 1])
    page_slice_b = entries[start:end]
    csv_bp = _export_csv(page_slice_b, '', filter_info_b)
    re1.download_button("📄 本页CSV", data=csv_bp,
                        file_name=f"浏览_{scope}_第{page+1}页.csv",
                        mime="text/csv", use_container_width=True)
    txt_bp = _export_txt(page_slice_b, '', filter_info_b)
    re2.download_button("📄 本页TXT", data=txt_bp,
                        file_name=f"浏览_{scope}_第{page+1}页.txt",
                        mime="text/plain", use_container_width=True)
    csv_ba = _export_csv(entries, '', filter_info_b)
    re3.download_button("📦 全部CSV", data=csv_ba,
                        file_name=f"浏览_{scope}.csv",
                        mime="text/csv", use_container_width=True)
    txt_ba = _export_txt(entries, '', filter_info_b)
    re4.download_button("📦 全部TXT", data=txt_ba,
                        file_name=f"浏览_{scope}.txt",
                        mime="text/plain", use_container_width=True)

    browse_sel_set = st.session_state[browse_sel_key]
    if browse_sel_set:
        st.markdown("**选中导出**:")
        browse_sel_rel = st.checkbox("包含关联条目", key=f"{browse_sel_key}_include_rel", value=False)
        col_s1, col_s2 = st.columns([1, 1])
        browse_sel_results = [entries[i] for i in browse_sel_set if i < len(entries)]
        csv_bs = _export_csv(browse_sel_results, '', filter_info_b, browse_sel_rel)
        col_s1.download_button(f"✅ 选中({len(browse_sel_set)})CSV", data=csv_bs,
                               file_name=f"浏览_{scope}_选中.csv",
                               mime="text/csv", use_container_width=True)
        txt_bs = _export_txt(browse_sel_results, '', filter_info_b, browse_sel_rel)
        col_s2.download_button(f"✅ 选中({len(browse_sel_set)})TXT", data=txt_bs,
                               file_name=f"浏览_{scope}_选中.txt",
                               mime="text/plain", use_container_width=True)

    # 显示条目
    for i_offset, e in enumerate(entries[start:end], start + 1):
        bidx = start + i_offset - 1
        ds = _fmt_date(e)
        wt = f" ☁️ {e['weather']}" if e.get('weather') else ""
        cat_badge = ""
        if e.get('category'):
            cat_badge = f" [{e['category']}"
            if e.get('sub_tag'):
                cat_badge += f" / {e['sub_tag']}"
            cat_badge += "]"
        col_a, col_b = st.columns([0.04, 0.96])
        with col_a:
            st.checkbox("##", key=f"bsel_{scope}_{bidx}", label_visibility="collapsed",
                        value=bidx in browse_sel_set, on_change=toggle_browse_sel, args=(bidx,))
        with col_b:
            with st.expander(f"[{i_offset}] 《{e['diary_name']}》 {ds}{wt}{cat_badge}", expanded=True):
                st.markdown(e['text'])
                st.caption(f"📖 {e['diary_name']} | {ds}{cat_badge}")
                # 关联条目
                rel = get_related(e['diary_name'], e.get('year'), e.get('month'), e.get('day'),
                                  (e.get('text') or '')[:50], e.get('category'), e.get('sub_tag'))
                if rel:
                    st.markdown("**🔗 关联条目**")
                    rel_key = f"rel_browse_{scope}_{start + i_offset - 1}"
                    cols_rel = st.columns(len(rel))
                    for j, (_, re) in enumerate(rel):
                        rds = _fmt_date(re)
                        short = f"{rds}" if rds else f"{re.get('year','')}"
                        if cols_rel[j].button(short, key=f"{rel_key}_{j}", use_container_width=True):
                            st.session_state[f"{rel_key}_show"] = j
                    sj = st.session_state.get(f"{rel_key}_show")
                    if sj is not None and sj < len(rel):
                        _, re_show = rel[sj]
                        st.markdown(f"> **《{re_show['diary_name']}》 {_fmt_date(re_show)}**")
                        st.markdown(re_show['text'])
                    if st.button("✕ 收起", key=f"{rel_key}_close"):
                        st.session_state[f"{rel_key}_show"] = None
                        st.rerun()

    # 底部翻页
    st.divider()
    c1, c2, c3 = st.columns([4, 1, 1])
    if c2.button("◀ 上一页", disabled=(page == 0), key="b_b2", use_container_width=True):
        st.session_state["browse_page"] = max(0, page - 1)
        st.rerun()
    if c3.button("下一页 ▶", disabled=(page >= total_pages - 1), key="b_b3", use_container_width=True):
        st.session_state["browse_page"] = min(total_pages - 1, page + 1)
        st.rerun()
    c1.markdown(f"<div style='text-align:right;padding:8px'>第 {page+1}/{total_pages} 页（共 {total} 条）</div>",
                unsafe_allow_html=True)

    st.stop()


# =========================================================================
# 情感分析
# =========================================================================
elif mode == "💬 情感分析":
    import altair as alt
    import pandas as pd
    import json
    import numpy as np
    import pickle as _pickle

    SENTIMENT_FILE = os.path.join(BASE_DIR, "parsed_data/sentiment_results.json")
    META_PATH = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")

    if not os.path.exists(SENTIMENT_FILE):
        st.warning("请先运行 sentiment_analysis.py")
        st.stop()

    with open(SENTIMENT_FILE) as f:
        sa_data = json.load(f)

    stats = sa_data['stats']
    entries_list = sa_data['entries']

    # Read sidebar widgets
    sa_view = st.session_state.get("sa_view", "情感演化")
    sa_sel_cat = st.session_state.get("sa_cat")
    sa_min_y = st.session_state.get("sa_min_y")
    sa_max_y = st.session_state.get("sa_max_y")

    # Load meta
    with open(META_PATH, 'rb') as f:
        meta = _pickle.load(f)

    # Filter indices
    filtered_indices = []
    for i, s_entry in enumerate(entries_list):
        e = meta[i] if i < len(meta) else {}
        yr = e.get('year')
        cat = e.get('category', '未分类')
        if sa_min_y and (yr is None or yr < sa_min_y): continue
        if sa_max_y and (yr is None or yr > sa_max_y): continue
        if sa_sel_cat and cat != sa_sel_cat: continue
        filtered_indices.append(i)

    if not filtered_indices and sa_view != "情感演化":
        st.info("没有符合条件的条目")
        st.stop()

    f_scores = [entries_list[i]['score'] for i in filtered_indices]
    f_meta = [meta[i] for i in filtered_indices]

    st.subheader("💬 情感分析")

    # ================================================================
    # 总体分布
    # ================================================================
    if sa_view == "总体分布":
        df_dist = pd.DataFrame({'score': f_scores})
        hist = alt.Chart(df_dist).mark_bar().encode(
            x=alt.X('score:Q', bin=alt.Bin(maxbins=40), title='情感得分'),
            y=alt.Y('count():Q', title='条目数'),
            tooltip=['count()'],
        ).properties(height=350, title="情感得分分布")
        rule = alt.Chart(pd.DataFrame({'mean': [stats['mean']]})).mark_rule(color='red', strokeDash=[5, 5]).encode(
            x='mean:Q'
        )
        st.altair_chart(hist + rule, use_container_width=True)
        st.caption(f"红色虚线为总体均值 ({stats['mean']:.4f})")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("平均分", f"{stats['mean']:.4f}")
        col2.metric("标准差", f"{stats['std']:.4f}")
        col3.metric("😊 正向", f"{stats['positive_pct']}%")
        col4.metric("😐 中立", f"{stats['neutral_pct']}%")
        col5.metric("😟 负向", f"{stats['negative_pct']}%")

    # ================================================================
    # 按年份趋势
    # ================================================================
    elif sa_view == "按年份趋势":
        by_year = sa_data['by_year']
        years_sorted = sorted([int(y) for y in by_year.keys()])
        records = []
        for y in years_sorted:
            d = by_year[str(y)]
            records.append({
                'year': y, 'mean': d['mean'], 'std': d['std'],
                'count': d['count'],
                'upper': d['mean'] + d['std'],
                'lower': d['mean'] - d['std'],
            })
        df_trend = pd.DataFrame(records)
        if sa_min_y: df_trend = df_trend[df_trend['year'] >= sa_min_y]
        if sa_max_y: df_trend = df_trend[df_trend['year'] <= sa_max_y]

        base = alt.Chart(df_trend).encode(x=alt.X('year:O', title='年份'))
        band = base.mark_area(opacity=0.15, color='#4c78a8').encode(
            y=alt.Y('lower:Q', title='情感得分'),
            y2='upper:Q',
        )
        line = base.mark_line(color='#4c78a8', point=True).encode(
            y=alt.Y('mean:Q', title='平均情感得分'),
            tooltip=['year', 'mean', 'std', 'count'],
        )
        st.altair_chart(band + line, use_container_width=True)
        st.caption("浅色区域为 ±1 标准差，点线为均值")

        # 情感强度叠加（v2）
        if 'intensity_by_year' in sa_data:
            st.divider()
            st.subheader("📊 情感强度趋势")
            st.caption("强度（0~1）衡量情感表达的激烈程度，越高说明该年情感词越密集")
            int_years = sorted(sa_data['intensity_by_year'].keys(), key=int)
            int_records = []
            for y_str in int_years:
                d = sa_data['intensity_by_year'][y_str]
                yr = int(y_str)
                if sa_min_y and yr < sa_min_y: continue
                if sa_max_y and yr > sa_max_y: continue
                int_records.append({
                    'year': yr, 'mean_intensity': d['mean_intensity'],
                    'max_intensity': d['max_intensity'], 'count': sa_data['by_year'][y_str]['count'],
                })
            if int_records:
                df_int = pd.DataFrame(int_records)
                int_line = alt.Chart(df_int).mark_line(color='#e74c3c', point=True, strokeWidth=3).encode(
                    x=alt.X('year:O', title='年份'),
                    y=alt.Y('mean_intensity:Q', title='平均强度', scale=alt.Scale(domain=[0, 0.8])),
                    tooltip=['year', 'mean_intensity', 'max_intensity', 'count'],
                )
                max_line = alt.Chart(df_int).mark_line(color='#e74c3c', strokeWidth=1, strokeDash=[5, 5], opacity=0.5).encode(
                    x='year:O', y='max_intensity:Q',
                )
                st.altair_chart((int_line + max_line).properties(height=300), use_container_width=True)
                st.caption("实线 = 年均强度 | 虚线 = 年最高强度")

            # 强度 vs 得分
            st.divider()
            st.subheader("🎯 情感强度与得分的关系")
            df_scatter = pd.DataFrame({
                'score': [entries_list[i]['score'] for i in filtered_indices],
                'intensity': [entries_list[i].get('intensity', abs(entries_list[i]['score'])) for i in filtered_indices],
            })
            hex_plot = alt.Chart(df_scatter).mark_rect().encode(
                x=alt.X('score:Q', bin=alt.Bin(maxbins=30), title='情感得分'),
                y=alt.Y('intensity:Q', bin=alt.Bin(maxbins=30), title='情感强度'),
                color=alt.Color('count():Q', scale=alt.Scale(scheme='reds')),
            ).properties(height=350)
            st.altair_chart(hex_plot, use_container_width=True)
            st.caption("颜色越深 = 该区间条目越多。右下（正向且强烈）≈ 胜利喜悦，左上（负向且强烈）≈ 惨烈战斗")

            # 高强度条目
            st.divider()
            st.subheader("🔥 高强度情感条目 Top 10")
            intense_idx = [(entries_list[i].get('intensity', 0), i, meta[i]) for i in filtered_indices]
            intense_idx.sort(key=lambda x: -x[0])
            for intensity, idx, e in intense_idx[:10]:
                ds = _fmt_date(e)
                score = entries_list[idx]['score']
                st.markdown(f"> **强度 {intensity:.4f}** | 得分 {score:+.4f} | 《{e['diary_name']}》 {ds}")
                st.markdown((e.get('text') or '')[:300])
                words = entries_list[idx].get('pos_words', [])[:4] + entries_list[idx].get('neg_words', [])[:4]
                if words:
                    st.caption(f"命中词: {'、'.join(words)}")
                st.divider()

        st.divider()
        st.subheader("📋 逐年统计数据")
        st.dataframe(df_trend.style.format({'mean': '{:.4f}', 'std': '{:.4f}', 'upper': '{:.4f}', 'lower': '{:.4f}'}),
                     use_container_width=True, hide_index=True)

    # ================================================================
    # 按分类对比
    # ================================================================
    elif sa_view == "按分类对比":
        by_cat = sa_data['by_category']
        cat_records = []
        for cat, d in by_cat.items():
            if cat in ('未分类', ''): continue
            cat_records.append({
                'category': cat, 'mean': d['mean'], 'std': d['std'],
                'count': d['count'], 'upper': d['mean'] + d['std'], 'lower': d['mean'] - d['std'],
            })
        df_cat = pd.DataFrame(cat_records)

        bars = alt.Chart(df_cat).mark_bar().encode(
            x=alt.X('mean:Q', title='平均情感得分'),
            y=alt.Y('category:N', title=None, sort='-x'),
            color=alt.Color('category:N', legend=None,
                           scale=alt.Scale(domain=['日常生活', '军事作战', '组织建设', '群众运动', '政权建设', '文化建设'],
                                           range=['#4a90d9', '#e74c3c', '#e8783a', '#50b86c', '#9b59b6', '#e67e22'])),
            tooltip=['category', 'mean', 'std', 'count'],
        ).properties(height=200)
        error_bars = alt.Chart(df_cat).mark_errorbar(extent='ci').encode(
            x=alt.X('mean:Q', title='平均情感得分'),
            y=alt.Y('category:N'),
        )
        st.altair_chart(bars + error_bars, use_container_width=True)
        st.caption("误差线为 95% 置信区间")

        st.dataframe(df_cat.style.format({'mean': '{:.4f}', 'std': '{:.4f}'}),
                     use_container_width=True, hide_index=True)

    # ================================================================
    # 按日记本对比
    # ================================================================
    elif sa_view == "按日记本对比":
        by_diary = sa_data['by_diary']
        diary_records = []
        for diary, d in by_diary.items():
            diary_records.append({
                'diary': diary[:25], 'mean': d['mean'], 'std': d['std'],
                'count': d['count'],
            })
        df_diary = pd.DataFrame(diary_records)
        df_diary = df_diary.sort_values('mean', ascending=True)

        bars = alt.Chart(df_diary).mark_bar().encode(
            x=alt.X('mean:Q', title='平均情感得分'),
            y=alt.Y('diary:N', title=None, sort='-x'),
            color=alt.condition(
                alt.datum.mean > 0,
                alt.value('#54a24b'),
                alt.value('#e45756')
            ),
            tooltip=['diary', 'mean', 'std', 'count'],
        ).properties(height=400)
        st.altair_chart(bars, use_container_width=True)
        st.caption("绿色 = 正向平均，红色 = 负向平均")

        st.dataframe(df_diary.style.format({'mean': '{:.4f}', 'std': '{:.4f}'}),
                     use_container_width=True, hide_index=True)

    # ================================================================
    # 情感演化（逐日 + 事件提取 + 年均趋势 + 交互分析）
    # ================================================================
    elif sa_view == "情感演化":
        sa_diary = st.session_state.get("sa_diary")
        include_national = st.session_state.get("sa_natl_events", True)

        if not sa_diary:
            st.info("请在侧边栏选择一本日记")
            st.stop()

        # 取该日记条目
        diary_indices = []
        diary_dates = []
        for i, e in enumerate(meta):
            if e.get('diary_name') == sa_diary:
                yr, mo, dy = e.get('year'), e.get('month'), e.get('day')
                if yr and mo and dy:
                    diary_indices.append(i)
                    diary_dates.append((yr, mo, dy))
                elif yr:
                    diary_indices.append(i)
                    diary_dates.append((yr, 6, 15))

        if not diary_indices:
            st.warning("该日记无有效日期数据")
            st.stop()

        # 构建 DataFrame（容错无效日期）
        valid_rows = []
        for idx, (y, m, d) in zip(diary_indices, diary_dates):
            try:
                dt = pd.Timestamp(f"{y}-{m:02d}-{d:02d}")
                intensity = entries_list[idx].get('intensity', abs(entries_list[idx]['score']))
                valid_rows.append({'date': dt, 'year': y, 'score': entries_list[idx]['score'], 'intensity': intensity})
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                continue

        if not valid_rows:
            st.warning("该日记无有效日期数据")
            st.stop()

        df_evo = pd.DataFrame(valid_rows).sort_values('date')
        evo_min_date = df_evo['date'].min()
        evo_max_date = df_evo['date'].max()

        st.subheader(f"📈 《{sa_diary}》情感演化")

        # ---- 事件提取 ----
        # 1) 从日记文本提取事件
        def extract_diary_events():
            import re
            named_pat = re.compile(r'[^，。；\s]{2,8}(?:战役|战斗|会战|大战|事变|起义|暴动|会师|会议|运动)')
            action_pat = re.compile(r'(?:攻克|收复|解放|占领|攻占|夺取|歼灭|击溃|围攻|突破|进驻|召开|组织|开展)(?:了)?[^，。；\s]{2,6}')
            sig_kw = ["总攻", "决战", "大捷", "告捷", "凯旋", "胜利会师",
                      "突围", "反攻", "全线出击", "扫荡", "围剿", "偷袭"]
            found = []
            seen = set()
            for i, e in enumerate(meta):
                if e.get('diary_name') != sa_diary:
                    continue
                yr, mo, dy = e.get('year'), e.get('month'), e.get('day')
                if not (yr and mo and dy):
                    continue
                text = e.get('text', '')
                try:
                    date_obj = pd.Timestamp(f"{yr}-{mo:02d}-{dy:02d}")
                except Exception:
                    continue
                if not (evo_min_date <= date_obj <= evo_max_date):
                    continue
                for m in named_pat.finditer(text):
                    name = m.group().strip()
                    if name not in seen and len(name) <= 12:
                        seen.add(name)
                        found.append((date_obj, name))
                for m in action_pat.finditer(text):
                    name = m.group().strip()
                    if name not in seen:
                        seen.add(name)
                        found.append((date_obj, name))
                for kw in sig_kw:
                    if kw in text and kw not in seen:
                        seen.add(kw)
                        for m in re.finditer(rf'[^。；\n]{{0,15}}{kw}[^。；\n]{{0,15}}', text):
                            snippet = m.group().strip()
                            if len(snippet) >= 4 and snippet not in seen:
                                seen.add(snippet)
                                found.append((date_obj, snippet[:30]))
                            break
            found.sort(key=lambda x: x[0])
            merged = []
            for date, name in found:
                if merged and merged[-1][1] == name:
                    continue
                if merged and (date - merged[-1][0]).days <= 7:
                    continue
                merged.append((date, name))
            return merged[:40]

        extracted_events = extract_diary_events()

        # 2) 国家级历史事件
        national_events = [
            ("1921-07-01", "中共成立"), ("1927-08-01", "南昌起义"),
            ("1931-09-18", "九一八事变"), ("1934-10-01", "长征开始"),
            ("1935-01-15", "遵义会议"), ("1936-10-01", "长征胜利"),
            ("1936-12-12", "西安事变"), ("1937-07-07", "全面抗战爆发"),
            ("1937-09-25", "平型关大捷"), ("1938-03-23", "台儿庄战役"),
            ("1940-08-20", "百团大战"), ("1941-01-04", "皖南事变"),
            ("1942-02-01", "整风运动"), ("1945-08-15", "抗战胜利"),
            ("1946-06-01", "全面内战"), ("1947-10-10", "土地改革"),
            ("1948-09-12", "三大战役"), ("1949-04-21", "渡江战役"),
            ("1949-10-01", "新中国成立"),
        ]

        # 合并全部事件
        all_events = list(extracted_events)
        if include_national:
            for date_str, label in national_events:
                evt_date = pd.Timestamp(date_str)
                if evo_min_date <= evt_date <= evo_max_date:
                    if not any(abs((evt_date - d).days) <= 15 and label[:2] in n for d, n in all_events):
                        all_events.append((evt_date, label))
        all_events.sort(key=lambda x: x[0])

        # 过滤掉前后7天无日记条目的事件
        def _has_entries_near(evt_date):
            for i in diary_indices:
                e = meta[i]
                yr, mo, dy = e.get('year'), e.get('month'), e.get('day')
                if not (yr and mo and dy):
                    continue
                try:
                    entry_date = pd.Timestamp(f"{yr}-{mo:02d}-{dy:02d}")
                except Exception:
                    continue
                if abs((entry_date - evt_date).days) <= 7:
                    return True
            return False

        all_events = [(d, l) for d, l in all_events if _has_entries_near(d)]

        # ---- 图表 ----
        # 逐年均线（加粗显示年份波动）
        yearly_mean = df_evo.groupby(df_evo['date'].dt.year)['score'].mean().reset_index()
        yearly_mean['mid_date'] = yearly_mean['date'].apply(lambda y: pd.Timestamp(f"{int(y)}-07-01"))

        # 按年聚合事件（避免图表上标记过多）
        from collections import defaultdict
        evt_by_year = defaultdict(list)
        for d, l in all_events:
            evt_by_year[d.year].append(l)
        yr_evt_rows = []
        for y, names in evt_by_year.items():
            yr_row = yearly_mean[yearly_mean['date'] == y]
            if not yr_row.empty:
                yr_evt_rows.append({
                    'date': pd.Timestamp(f"{y}-07-01"),
                    'mean': yr_row.iloc[0]['score'],
                    'events': '、'.join(names[:5]) + (f' 等{len(names)}个' if len(names) > 5 else ''),
                })

        scatter = alt.Chart(df_evo).mark_circle(size=10, opacity=0.12, color='#4c78a8').encode(
            x=alt.X('date:T', title='日期'),
            y=alt.Y('score:Q', title='情感得分', scale=alt.Scale(domain=[-1, 1])),
        )
        yr_line = alt.Chart(yearly_mean).mark_line(color='#2c3e50', strokeWidth=4, opacity=0.85).encode(
            x='mid_date:T', y='score:Q',
        )
        yr_dots = alt.Chart(yearly_mean).mark_point(size=110, color='#2c3e50', filled=True, opacity=0.85).encode(
            x='mid_date:T', y='score:Q',
            tooltip=[alt.Tooltip('date:Q', title='年份'), alt.Tooltip('score:Q', title='年均分', format='.4f')],
        )
        zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3, 3], opacity=0.5).encode(
            y='y:Q'
        )
        chart = scatter + zero + yr_line + yr_dots

        if yr_evt_rows:
            evt_m_df = pd.DataFrame(yr_evt_rows)
            evt_markers = alt.Chart(evt_m_df).mark_point(size=90, color='red', shape='triangle-down', filled=True).encode(
                x='date:T', y='mean:Q', tooltip=['events:N'],
            )
            chart = chart + evt_markers

        st.altair_chart(chart.properties(height=450), use_container_width=True)
        st.caption("蓝色散点 = 逐条日记 | **黑色粗线 = 年均值** | 红色 ▼ = 事件（悬停查看）")

        # ---- 情感强度趋势（第二张图） ----
        intensity_scatter = alt.Chart(df_evo).mark_circle(size=8, opacity=0.08, color='#e74c3c').encode(
            x=alt.X('date:T', title='日期'),
            y=alt.Y('intensity:Q', title='情感强度', scale=alt.Scale(domain=[0, 1])),
        )
        yearly_intensity = df_evo.groupby(df_evo['date'].dt.year)['intensity'].mean().reset_index()
        yearly_intensity['mid_date'] = yearly_intensity['date'].apply(lambda y: pd.Timestamp(f"{int(y)}-07-01"))
        int_line = alt.Chart(yearly_intensity).mark_line(color='#e74c3c', strokeWidth=3, opacity=0.7).encode(
            x='mid_date:T', y='intensity:Q',
        )
        int_chart = intensity_scatter + int_line
        st.altair_chart(int_chart.properties(height=200), use_container_width=True)
        st.caption("**红色 = 情感强度趋势**（越强 = 情感表达越激烈，越高说明该年情感词密度大）")

        # ---- 情感子类分布（仅文本条） ----
        if 'emotion_profile' in entries_list[diary_indices[0]]:
            st.divider()
            st.subheader("🎭 情感子类分布")
            emotion_totals = {cat: 0 for cat in ['喜悦','悲伤','愤怒','恐惧','期盼','信任']}
            for i in diary_indices:
                ep = entries_list[i].get('emotion_profile', {})
                for cat in emotion_totals:
                    emotion_totals[cat] += ep.get(cat, 0)
            total_emo = sum(emotion_totals.values()) or 1
            emotion_pcts = {k: round(v / total_emo * 100, 1) for k, v in emotion_totals.items()}
            for cat, pct in sorted(emotion_pcts.items(), key=lambda x: -x[1]):
                bar_len = max(1, int(pct / 2))
                bar = '█' * bar_len
                st.markdown(f"**{cat}** {pct}% {bar}")

        # ---- 历史事件交互区（全部事件可选） ----
        st.divider()
        st.subheader("📋 历史事件与日记对照")

        if all_events:
            event_options = [f"{l}（{d.strftime('%Y-%m-%d')}）" for d, l in all_events]
            sel_idx = st.selectbox("选择事件，查看前后30天情感变化及相关日记",
                                   range(len(event_options)),
                                   format_func=lambda i: event_options[i])
            evt_date, evt_name = all_events[sel_idx]

            before = df_evo[(df_evo['date'] >= evt_date - pd.Timedelta(days=30)) & (df_evo['date'] < evt_date)]['score']
            after = df_evo[(df_evo['date'] >= evt_date) & (df_evo['date'] <= evt_date + pd.Timedelta(days=30))]['score']
            b_mean = before.mean() if len(before) > 0 else None
            a_mean = after.mean() if len(after) > 0 else None
            change = (a_mean - b_mean) if (b_mean is not None and a_mean is not None) else None

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("事件前30天均分", f"{b_mean:.4f}" if b_mean is not None else '—')
            col_m2.metric("事件后30天均分", f"{a_mean:.4f}" if a_mean is not None else '—')
            col_m3.metric("变化", f"{change:+.4f}" if change is not None else '—', delta_color="off")

            st.markdown(f"**📖 「{evt_name}」前后7天的日记条目**")
            related = []
            for i in diary_indices:
                e = meta[i]
                yr, mo, dy = e.get('year'), e.get('month'), e.get('day')
                if not (yr and mo and dy):
                    continue
                try:
                    entry_date = pd.Timestamp(f"{yr}-{mo:02d}-{dy:02d}")
                except Exception:
                    continue
                if (evt_date - pd.Timedelta(days=7)) <= entry_date <= (evt_date + pd.Timedelta(days=7)):
                    related.append((entry_date, i, e))
            related.sort(key=lambda x: x[0])

            if related:
                for entry_date, idx, e in related:
                    score = entries_list[idx]['score']
                    neg_words = entries_list[idx].get('neg_words', [])
                    pos_words = entries_list[idx].get('pos_words', [])
                    lbl = f"{entry_date.strftime('%Y-%m-%d')}  情感:{score:+.4f}"
                    if pos_words:
                        lbl += f"  😊{'、'.join(pos_words[:4])}"
                    if neg_words:
                        lbl += f"  😟{'、'.join(neg_words[:4])}"
                    with st.expander(lbl):
                        st.markdown(e.get('text', ''))
            else:
                st.info("该事件前后7天无日记条目")
        else:
            st.info("该日记时间范围内无对应历史事件")

        # 日记统计摘要
        st.divider()
        st.subheader("📊 日记情感摘要")
        col1, col2, col3, col4, col5 = st.columns(5)
        diary_scores = [entries_list[i]['score'] for i in diary_indices]
        col1.metric("条目数", f"{len(diary_scores)}")
        col2.metric("平均情感", f"{np.mean(diary_scores):.4f}")
        col3.metric("标准差", f"{np.std(diary_scores):.4f}")
        col4.metric("正向占比", f"{np.mean(np.array(diary_scores) > 0.05)*100:.1f}%")
        col5.metric("负向占比", f"{np.mean(np.array(diary_scores) < -0.05)*100:.1f}%")
        st.caption(f"时间范围: {evo_min_date.strftime('%Y-%m-%d')} ~ {evo_max_date.strftime('%Y-%m-%d')}")

        # 该日记极端条目
        st.divider()
        st.subheader("📋 该日记极端条目")
        diary_scored = [(entries_list[i]['score'], i, meta[i]) for i in diary_indices]
        diary_scored.sort(key=lambda x: x[0])
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**😟 最负面 (Top 5)**")
            for score, idx, e in diary_scored[:5]:
                ds = _fmt_date(e)
                st.markdown(f"> **[{score:.4f}]** {ds}")
                st.markdown((e.get('text') or '')[:300])
                words = entries_list[idx].get('neg_words', [])
                if words:
                    st.caption(f"负向词: {'、'.join(words[:6])}")
        with col_e2:
            st.markdown("**😊 最正面 (Top 5)**")
            for score, idx, e in reversed(diary_scored[-5:]):
                ds = _fmt_date(e)
                st.markdown(f"> **[{score:.4f}]** {ds}")
                st.markdown((e.get('text') or '')[:300])
                words = entries_list[idx].get('pos_words', [])
                if words:
                    st.caption(f"正向词: {'、'.join(words[:6])}")

        # 该日记极端条目
        st.divider()
        st.subheader("📋 该日记极端条目")
        diary_scored = [(entries_list[i]['score'], i, meta[i]) for i in diary_indices]
        diary_scored.sort(key=lambda x: x[0])
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**😟 最负面 (Top 5)**")
            for score, idx, e in diary_scored[:5]:
                ds = _fmt_date(e)
                st.markdown(f"> **[{score:.4f}]** {ds}")
                st.markdown((e.get('text') or '')[:300])
                words = entries_list[idx].get('neg_words', [])
                if words:
                    st.caption(f"负向词: {'、'.join(words[:6])}")
        with col_e2:
            st.markdown("**😊 最正面 (Top 5)**")
            for score, idx, e in reversed(diary_scored[-5:]):
                ds = _fmt_date(e)
                st.markdown(f"> **[{score:.4f}]** {ds}")
                st.markdown((e.get('text') or '')[:300])
                words = entries_list[idx].get('pos_words', [])
                if words:
                    st.caption(f"正向词: {'、'.join(words[:6])}")

        # ---- 情感转折点 ----
        st.divider()
        st.subheader("🔄 情感转折点")
        diary_tps = [tp for tp in sa_data.get('turning_points', [])
                     if evo_min_date <= pd.Timestamp(tp['date']) <= evo_max_date]
        if diary_tps:
            st.markdown(f"检测到 **{len(diary_tps)}** 个情感转折点")
            tp_df = pd.DataFrame(diary_tps)
            tp_df['date'] = pd.to_datetime(tp_df['date'])
            tp_chart = alt.Chart(tp_df).mark_point(size=100, filled=True, opacity=0.85).encode(
                x=alt.X('date:T', title='日期'),
                y=alt.Y('mean_after:Q', title='转折后均分', scale=alt.Scale(domain=[-1, 1])),
                color=alt.condition(
                    alt.datum.type == 'up',
                    alt.value('#27ae60'),
                    alt.value('#c0392b')
                ),
                shape=alt.condition(
                    alt.datum.type == 'up',
                    alt.value('triangle-up'),
                    alt.value('triangle-down')
                ),
                tooltip=['date:T', 'change:Q', 'mean_before:Q', 'mean_after:Q'],
            ).properties(height=200)
            st.altair_chart(tp_chart, use_container_width=True)
            with st.expander("📋 转折点详情"):
                for tp in diary_tps[:20]:
                    direction = "📈" if tp['type'] == 'up' else "📉"
                    st.markdown(
                        f"> **{tp['date']}** {direction} "
                        f"变化 {tp['change']:+.4f} "
                        f"(前 {tp['mean_before']:.4f} → 后 {tp['mean_after']:.4f})"
                    )
        else:
            st.info("该日记时间范围内未检测到显著情感转折点")

    # ================================================================
    # 极端案例
    # ================================================================
    elif sa_view == "极端案例":
        scored = [(entries_list[i]['score'], i, meta[i]) for i in filtered_indices]
        scored.sort(key=lambda x: x[0])

        st.subheader("😟 最负面条目 (Top 10)")
        for score, idx, e in scored[:10]:
            ds = _fmt_date(e)
            st.markdown(f"> **[{score:.4f}]** 《{e['diary_name']}》 {ds}")
            st.markdown((e.get('text') or '')[:500])
            words = entries_list[idx].get('neg_words', [])
            if words:
                st.caption(f"命中负向词: {'、'.join(words[:8])}")
            st.divider()

        st.subheader("😊 最正面条目 (Top 10)")
        for score, idx, e in reversed(scored[-10:]):
            ds = _fmt_date(e)
            st.markdown(f"> **[{score:.4f}]** 《{e['diary_name']}》 {ds}")
            st.markdown((e.get('text') or '')[:500])
            words = entries_list[idx].get('pos_words', [])
            if words:
                st.caption(f"命中正向词: {'、'.join(words[:8])}")
            st.divider()

    # ================================================================
    # 情感画像（新增 v2 分析）
    # ================================================================
    elif sa_view == "情感画像":
        _v2_available = 'emotion_by_year' in sa_data
        if not _v2_available:
            st.info("请先运行增强版 sentiment_analysis.py 以获取情感画像数据")
            st.stop()

        st.subheader("🎭 情感画像分析")
        st.caption("基于6类情感子类（喜悦、悲伤、愤怒、恐惧、期盼、信任）分析革命日记的情感结构")

        # 按分类的情感画像
        if 'emotion_profile' in entries_list[0]:
            st.markdown("**各分类的情感子类分布**")
            cat_emo_data = {}
            for cat, cat_stats in sa_data['by_category'].items():
                if cat in ('未分类', ''):
                    continue
                cat_emo_data[cat] = {'喜悦': 0, '悲伤': 0, '愤怒': 0, '恐惧': 0, '期盼': 0, '信任': 0}
                cat_count = 0
                for i in filtered_indices:
                    e = meta[i] if i < len(meta) else {}
                    if e.get('category', '未分类') == cat:
                        ep = entries_list[i].get('emotion_profile', {})
                        for emo_cat in cat_emo_data[cat]:
                            cat_emo_data[cat][emo_cat] += ep.get(emo_cat, 0)
                        cat_count += 1
                # 归一化
                if cat_count > 0:
                    for emo_cat in cat_emo_data[cat]:
                        cat_emo_data[cat][emo_cat] = round(cat_emo_data[cat][emo_cat] / cat_count, 4)

            emo_df_rows = []
            for cat, emo_dict in cat_emo_data.items():
                for emo_cat, val in emo_dict.items():
                    emo_df_rows.append({'category': cat, 'emotion': emo_cat, 'value': val})
            if emo_df_rows:
                emo_df = pd.DataFrame(emo_df_rows)
                heatmap = alt.Chart(emo_df).mark_rect().encode(
                    x=alt.X('category:N', title='分类', sort='-x'),
                    y=alt.Y('emotion:N', title='情感子类'),
                    color=alt.Color('value:Q', title='频次/条', scale=alt.Scale(scheme='reds')),
                    tooltip=['category', 'emotion', 'value'],
                ).properties(height=250)
                st.altair_chart(heatmap, use_container_width=True)
                st.caption("颜色越深 = 该分类中此类情感词出现越频繁")

        # 年代情感画像热力图
        st.divider()
        st.markdown("**年代情感画像热力图**")
        emo_years = sorted(sa_data.get('emotion_by_year', {}).keys(), key=int)
        if emo_years:
            emo_yr_rows = []
            for y_str in emo_years:
                yr = int(y_str)
                if sa_min_y and yr < sa_min_y: continue
                if sa_max_y and yr > sa_max_y: continue
                for emo_cat, val in sa_data['emotion_by_year'][y_str].items():
                    emo_yr_rows.append({'year': yr, 'emotion': emo_cat, 'value': val})
            if emo_yr_rows:
                emo_yr_df = pd.DataFrame(emo_yr_rows)
                yr_heatmap = alt.Chart(emo_yr_df).mark_rect().encode(
                    x=alt.X('year:O', title='年份'),
                    y=alt.Y('emotion:N', title='情感子类'),
                    color=alt.Color('value:Q', title='频次/条', scale=alt.Scale(scheme='blues')),
                    tooltip=['year', 'emotion', 'value'],
                ).properties(height=250)
                st.altair_chart(yr_heatmap, use_container_width=True)
                st.caption("各年份中每类情感词的平均出现频次")

        # 各日记情感画像对比
        st.divider()
        st.markdown("**各日记情感画像对比**")
        diary_list_sorted = sorted(sa_data['by_diary'].keys())
        sel_diary_for_profile = st.selectbox("选择日记查看情感画像", diary_list_sorted, key="sa_profile_diary")
        if sel_diary_for_profile:
            diary_indices_p = []
            for i, e in enumerate(meta):
                if e.get('diary_name') == sel_diary_for_profile:
                    diary_indices_p.append(i)
            if diary_indices_p:
                emo_totals = {cat: 0 for cat in ['喜悦','悲伤','愤怒','恐惧','期盼','信任']}
                for i in diary_indices_p:
                    ep = entries_list[i].get('emotion_profile', {})
                    for cat in emo_totals:
                        emo_totals[cat] += ep.get(cat, 0)
                total_emo = sum(emo_totals.values()) or 1
                emo_pcts = {k: round(v / total_emo * 100, 1) for k, v in emo_totals.items()}
                for cat, pct in sorted(emo_pcts.items(), key=lambda x: -x[1]):
                    bar_len = max(1, int(pct / 2))
                    bar = '█' * bar_len
                    st.markdown(f"**{cat}** {pct}% {bar}")

    st.stop()



# =========================================================================
# 事件脉络
# =========================================================================
elif mode == "🗺️ 事件脉络":
    EVENT_FILE = os.path.join(BASE_DIR, "parsed_data/event_timeline.json")
    if not os.path.exists(EVENT_FILE):
        st.warning("⚠️ 请先运行 event_extraction.py 生成事件数据")
        st.stop()

    ev_data = load_event_data(500, os.path.getmtime(EVENT_FILE))
    if not ev_data:
        st.warning("事件数据为空")
        st.stop()

    ev_sub_mode = st.session_state.get("ev_sub_mode", "事件脉络")

    if ev_sub_mode == "知识图谱":
        st.subheader("🔗 知识图谱")
        if "wc_data" in st.session_state:
            ss, wf = st.session_state["wc_data"]
            render_knowledge_graph(ss, wf)
        else:
            st.info("请在左侧栏选择子标签并点击「生成知识图谱」")
        st.stop()

    clusters = ev_data["event_clusters"]
    ev_min_imp = st.session_state.get("ev_min_imp", 0.3)
    ev_category = st.session_state.get("ev_category", "全部")

    # 筛选
    filtered = [c for c in clusters if c["importance"] >= ev_min_imp]
    if ev_category != "全部":
        filtered = [c for c in filtered if c.get("category", "其他") == ev_category]

    st.subheader("🗺️ 跨日记事件详情")
    st.caption(f"显示 {len(filtered)}/{len(clusters)} 个事件簇 | "
              f"{f'门类: {ev_category} | ' if ev_category != '全部' else ''}"
              f"跨日记 {ev_data['statistics']['cross_diary_event_count']} 个")

    render_cross_diary_detail(filtered)
    st.stop()


# =========================================================================
# 检索模式（默认）
# =========================================================================
# 分类体系说明
with st.expander("📂 分类体系说明", expanded=False):
    st.markdown("""
    | 大类 | 子标签 |
    |------|--------|
    | **日常生活** | 住宿、伙食、文娱、穿着、医疗、其他 |
    | **军事作战** | 行军、战斗、训练、侦察、武器装备、其他 |
    | **组织建设** | 会议、学习、发展党员、组织生活、其他 |
    | **群众运动** | 妇女、青年、农民、工人运动、其他 |
    | **政权建设** | 选举、税收、司法、行政管理、其他 |
    | **文化建设** | 报刊、宣传、图画、戏剧、歌咏、墙报、标语、其他 |
    """)

# 查询栏（自动与 session_state.search_input 双向同步）
query = st.text_input("💬 搜索", placeholder="如：长征中的粮食问题、军民关系、冬季行军...",
                      label_visibility="collapsed", key="search_input")

# 快捷分类浏览（点击即筛选对应分类+子标签）
st.markdown("**快捷分类浏览**:")
quick_map = [
    ("住宿条件", "日常生活", "住宿"),
    ("粮食困难", "日常生活", "伙食"),
    ("文化娱乐", "日常生活", "文娱"),
    ("伤员救治", "日常生活", "医疗"),
    ("冬季行军", "军事作战", "行军"),
    ("战斗场景", "军事作战", "战斗"),
    ("军民关系", "日常生活", "文娱"),
    ("思想学习", "组织建设", "学习"),
]

def set_quick_filter(cat, tag):
    st.session_state["cat_select"] = cat
    st.session_state["tag_select"] = tag
    st.session_state["search_input"] = ""

cols = st.columns(len(quick_map))
for i, (label, cat, tag) in enumerate(quick_map):
    cols[i].button(label, key=f"q_{i}",
                   on_click=set_quick_filter, args=(cat, tag),
                   use_container_width=True)

st.divider()

# 查询中的历史事件自动识别（如"长征中的粮食问题"→自动按1934-1936筛选）
EVENT_YEAR_MAP = {
    "长征": (1934, 1937),
    "红军长征": (1934, 1937),
    "长征开始": (1934, 1935),
    "长征胜利": (1936, 1937),
    "北伐战争": (1926, 1928),
    "抗日战争": (1931, 1945),
    "抗战": (1931, 1945),
    "解放战争": (1945, 1950),
    "抗美援朝": (1950, 1954),
    "百团大战": (1940, 1942),
    "西安事变": (1936, 1937),
    "七七事变": (1937, 1938),
    "南昌起义": (1927, 1928),
    "秋收起义": (1927, 1928),
    "遵义会议": (1935, 1936),
    "土地改革": (1947, 1950),
    "土改": (1947, 1950),
    "大生产运动": (1942, 1946),
    "整风运动": (1942, 1946),
    "大生产": (1942, 1946),
    "整风": (1942, 1946),
    "苏维埃运动": (1927, 1937),
    "土地革命战争": (1927, 1937),
    "解放": (1945, 1950),
}


def _detect_event_years(query):
    """从查询文本中识别历史事件，返回(min_year, max_year, event_name)或(None,None,None)"""
    if not query:
        return None, None, None
    for event, (y1, y2) in sorted(EVENT_YEAR_MAP.items(), key=lambda x: -len(x[0])):
        if event in query:
            return y1, y2, event
    return None, None, None


# 取查询词
q = (st.session_state.get("search_input", "") or "").strip()

# 是否启用了筛选（无搜索词时也可浏览）
has_filters = sel_cat or sel_tag or min_y or max_y or sel_diaries

if q or has_filters:
    engine = get_engine()
    if engine is None:
        st.stop()

    if api_key and use_llm:
        engine.save_api_key(api_key)

    # 构建筛选信息
    filter_parts = []
    if sel_cat:
        filter_parts.append(f"大类={sel_cat}")
        if sel_tag:
            filter_parts.append(f"子标签={sel_tag}")
    if min_y:
        filter_parts.append(f"≥{min_y}年")
    if max_y:
        filter_parts.append(f"≤{max_y}年")
    if sel_diaries:
        filter_parts.append(f"{len(sel_diaries)}本日记")
    filter_info = f" | {' '.join(filter_parts)}" if filter_parts else ""

    if q:
        # 从查询中自动识别历史事件时间段
        ev_min_y, ev_max_y, ev_name = _detect_event_years(q)
        # 仅当用户未手动设置年份范围时，才使用自动识别的时间段
        search_min_y = ev_min_y if min_y is None else min_y
        search_max_y = ev_max_y if max_y is None else max_y
        if ev_name:
            st.caption(f"📌 检测到历史事件「{ev_name}」，自动限定时间段 {ev_min_y}-{ev_max_y}（可在左侧栏调整）")

        # 搜索模式：关键词+筛选
        with st.spinner(f"搜索「{q}」{filter_info}..."):
            results = engine.search(
                q, top_k=500,
                min_year=search_min_y, max_year=search_max_y,
                diaries=sel_diaries if sel_diaries else None,
                use_expansion=(use_llm and api_key),
                category=sel_cat,
                sub_tag=sel_tag,
            )
    else:
        # 浏览模式：仅按筛选条件浏览全部条目
        with st.spinner(f"浏览{filter_info}..."):
            results = engine.browse(
                top_k=500,
                min_year=min_y, max_year=max_y,
                diaries=sel_diaries if sel_diaries else None,
                category=sel_cat,
                sub_tag=sel_tag,
            )

    if results:
        total = len(results)
        total_pages = (total + page_size - 1) // page_size

        # 初始化/重置页码
        if "page" not in st.session_state:
            st.session_state.page = 0

        # 如果查询或筛选变了，重置到第一页
        page_key = (q or "__browse__") + str(filter_parts)
        if st.session_state.get("last_page_key") != page_key:
            st.session_state.page = 0
            st.session_state.last_page_key = page_key

        page = st.session_state.page
        start = page * page_size
        end = min(start + page_size, total)

        # 分页导航
        col_info, col_prev, col_page, col_next, col_gap = st.columns([3, 1, 2, 1, 3])
        col_info.metric(f"找到 {total} 条{filter_info}", f"第{page+1}/{total_pages}页")
        if col_prev.button("◀ 上一页", disabled=(page == 0), use_container_width=True):
            st.session_state.page = max(0, page - 1)
            st.rerun()
        col_page.markdown(f"<div style='text-align:center;padding:8px'>{page+1} / {total_pages}</div>",
                          unsafe_allow_html=True)
        if col_next.button("下一页 ▶", disabled=(page >= total_pages - 1), use_container_width=True):
            st.session_state.page = min(total_pages - 1, page + 1)
            st.rerun()

        # 导出
        st.markdown("**导出**:")
        ec1, ec2, ec3, ec4 = st.columns([1, 1, 1, 1])
        # 当前页
        page_slice = results[start:end]
        csv_page = _export_csv(page_slice, q, filter_info)
        ec1.download_button("📄 本页CSV", data=csv_page,
                            file_name=f"搜索结果_{q or '浏览'}_第{page+1}页.csv",
                            mime="text/csv", use_container_width=True)
        txt_page = _export_txt(page_slice, q, filter_info)
        ec2.download_button("📄 本页TXT", data=txt_page,
                            file_name=f"搜索结果_{q or '浏览'}_第{page+1}页.txt",
                            mime="text/plain", use_container_width=True)
        # 全部
        csv_all = _export_csv(results, q, filter_info)
        ec3.download_button("📦 全部CSV", data=csv_all,
                            file_name=f"搜索结果_{q or '浏览'}.csv",
                            mime="text/csv", use_container_width=True)
        txt_all = _export_txt(results, q, filter_info)
        ec4.download_button("📦 全部TXT", data=txt_all,
                            file_name=f"搜索结果_{q or '浏览'}.txt",
                            mime="text/plain", use_container_width=True)

        # 选中条目导出（初始化选中集合）
        sel_key = "export_sel_" + str(st.session_state.get("last_page_key", ""))
        if sel_key not in st.session_state:
            st.session_state[sel_key] = set()

        def toggle_sel(idx):
            s = st.session_state[sel_key]
            if idx in s:
                s.remove(idx)
            else:
                s.add(idx)

        sel_set = st.session_state[sel_key]
        if sel_set:
            st.markdown("**选中导出**:")
            sel_include_rel = st.checkbox("包含关联条目", key="sel_include_rel", value=False)
            col_s1, col_s2 = st.columns([1, 1])
            sel_results = [results[i] for i in sel_set if i < len(results)]
            csv_sel = _export_csv(sel_results, q, filter_info + f" | 已选{len(sel_results)}条", sel_include_rel)
            col_s1.download_button(f"✅ 选中({len(sel_set)})CSV", data=csv_sel,
                                   file_name=f"搜索结果_{q or '浏览'}_选中.csv",
                                   mime="text/csv", use_container_width=True)
            txt_sel = _export_txt(sel_results, q, filter_info + f" | 已选{len(sel_results)}条", sel_include_rel)
            col_s2.download_button(f"✅ 选中({len(sel_set)})TXT", data=txt_sel,
                                   file_name=f"搜索结果_{q or '浏览'}_选中.txt",
                                   mime="text/plain", use_container_width=True)

        # 显示当前页结果（带选择框）
        for i_offset, r in enumerate(results[start:end], start + 1):
            idx = start + i_offset - 1  # 全局索引
            ds = f"{r['year']}年{r['month']}月{r['day']}日" if r.get('year') and r.get('month') and r.get('day') else (r.get('date_raw') or '')
            wt = f" ☁️ {r['weather']}" if r.get('weather') else ""

            cat_badge = ""
            if r.get('category'):
                cat_badge = f" [{r['category']}"
                if r.get('sub_tag'):
                    cat_badge += f" / {r['sub_tag']}"
                cat_badge += "]"

            col_a, col_b = st.columns([0.04, 0.96])
            with col_a:
                st.checkbox("##", key=f"sel_{idx}", label_visibility="collapsed",
                            value=idx in sel_set, on_change=toggle_sel, args=(idx,))
            with col_b:
                with st.expander(f"[{i_offset}] 《{r['diary_name']}》 {ds}{wt}{cat_badge}  ({r['score']*100:.0f}%)", expanded=i_offset - start <= 3):
                    st.markdown(r['text'])
                    st.caption(f"📖 {r['diary_name']} | {ds}{cat_badge} | 匹配度 {r['score']*100:.0f}%")
                    # 关联条目
                    rel = get_related(r['diary_name'], r.get('year'), r.get('month'), r.get('day'),
                                      (r.get('text') or '')[:50], r.get('category'), r.get('sub_tag'))
                    if rel:
                        st.markdown("**🔗 关联条目**")
                        rel_key = f"rel_{start + i_offset - 1}"
                        cols_rel = st.columns(len(rel))
                        for j, (_, re) in enumerate(rel):
                            rds = _fmt_date(re)
                            short = f"{rds}" if rds else f"{re.get('year','')}"
                            if cols_rel[j].button(short, key=f"{rel_key}_{j}", use_container_width=True):
                                st.session_state[f"{rel_key}_show"] = j
                        sj = st.session_state.get(f"{rel_key}_show")
                        if sj is not None and sj < len(rel):
                            _, re_show = rel[sj]
                            st.markdown(f"> **《{re_show['diary_name']}》 {_fmt_date(re_show)}**")
                            st.markdown(re_show['text'])
                        if st.button("✕ 收起", key=f"{rel_key}_close"):
                            st.session_state[f"{rel_key}_show"] = None
                            st.rerun()

        # 底部翻页
        st.divider()
        c1, c2, c3, c4 = st.columns([4, 1, 1, 4])
        if c2.button("◀ 上一页", disabled=(page == 0), key="search_prev", use_container_width=True):
            st.session_state.page = max(0, page - 1)
            st.rerun()
        if c3.button("下一页 ▶", disabled=(page >= total_pages - 1), key="search_next", use_container_width=True):
            st.session_state.page = min(total_pages - 1, page + 1)
            st.rerun()
        c1.markdown(f"<div style='text-align:right;padding:8px'>第{page+1}/{total_pages}页（共{total}条）</div>",
                     unsafe_allow_html=True)
    else:
        st.warning("未找到相关内容，换个关键词试试")
else:
    st.info("💡 输入关键词搜索，或在左侧筛选大类/子标签/年代/日记后点击搜索浏览全部。")
