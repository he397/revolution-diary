"""
革命日记事件脉络提取
====================
实体提取（事件、人物、地点）+ 事件链接 + 知识图谱数据输出

输出: parsed_data/event_timeline.json

使用:
  python3 event_extraction.py
"""

import os, re, json, pickle
import math
from datetime import datetime, timedelta
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(BASE_DIR, "search_index/entries_meta.pkl")
SENTIMENT_PATH = os.path.join(BASE_DIR, "parsed_data/sentiment_results.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "parsed_data/event_timeline.json")

# ============================================================
# 1. 领域实体词典
# ============================================================

KNOWN_PERSONS = frozenset([
    "毛泽东", "朱德", "周恩来", "刘少奇", "任弼时", "邓小平", "彭德怀",
    "刘伯承", "贺龙", "陈毅", "罗荣桓", "徐向前", "聂荣臻", "叶剑英",
    "陈云", "李先念", "粟裕", "徐海东", "黄克诚", "陈赓", "谭政",
    "萧劲光", "张云逸", "罗瑞卿", "王树声", "许光达", "林彪",
    "博古", "王明", "张国焘", "张闻天", "王稼祥",
    "蒋介石", "张学良", "杨虎城", "傅作义", "冯玉祥", "何应钦",
    "李宗仁", "白崇禧", "卫立煌", "杜聿明", "张自忠",
    "鲁迅", "宋庆龄", "郭沫若", "茅盾", "丁玲",
    "秦基伟", "王近山", "陈再道", "杨勇", "杨得志",
    "陈锡联", "王震", "萧克", "宋时轮",
    "左权", "叶挺", "项英", "方志敏", "刘志丹",
    "赵尚志", "杨靖宇", "赵一曼", "吉鸿昌",
    "董必武", "林伯渠", "吴玉章", "徐特立", "谢觉哉",
])

KNOWN_LOCATIONS = frozenset([
    "延安", "井冈山", "瑞金", "遵义", "西柏坡", "陕甘宁", "晋察冀",
    "晋冀鲁豫", "晋绥", "山东", "华中", "苏北", "苏南", "皖南", "皖北",
    "鄂豫皖", "湘鄂赣", "川陕", "大别山", "太行山", "沂蒙山", "南泥湾",
    "北平", "南京", "上海", "天津", "重庆", "广州", "武汉", "西安",
    "沈阳", "长春", "哈尔滨", "长沙", "南昌", "贵阳", "昆明", "成都",
    "兰州", "西宁", "银川", "乌鲁木齐", "呼和浩特",
    "黑龙江", "吉林", "辽宁", "热河", "察哈尔", "绥远", "宁夏",
    "甘肃", "陕西", "山西", "河北", "山东", "河南", "湖北", "湖南",
    "江西", "安徽", "江苏", "浙江", "福建", "广东", "广西", "贵州",
    "云南", "四川", "西康", "青海", "西藏",
    "卢沟桥", "台儿庄", "平型关", "娘子关", "山海关", "嘉峪关",
    "泸定桥", "安顺场", "腊子口", "娄山关",
])

KNOWN_ORGANIZATIONS = frozenset([
    "八路军", "新四军", "红军", "解放军", "东北联军", "志愿军",
    "红四军", "红一方面军", "红二方面军", "红四方面军", "红二十五军",
    "第一野战军", "第二野战军", "第三野战军", "第四野战军",
    "华北军区", "东北军区", "华东军区", "中南军区", "西北军区",
    "抗大", "抗日军政大学", "陕北公学", "鲁迅艺术学院",
    "边区政府", "参议会", "人民政府",
    "国民党", "共产党",
])

# 已知重大事件（精确匹配用）
KNOWN_EVENTS = frozenset([
    "百团大战", "平型关大捷", "平型关战斗", "台儿庄战役", "台儿庄大战",
    "遵义会议", "西安事变", "皖南事变", "七七事变", "九一八事变", "卢沟桥事变",
    "南昌起义", "秋收起义", "广州起义", "湘南起义", "海陆丰起义", "渭华起义",
    "整风运动", "大生产运动", "土地改革运动", "五四运动", "新文化运动",
    "古田会议", "八七会议", "中共一大", "中共七大", "瓦窑堡会议", "洛川会议",
    "湘江战役", "渡江战役", "平津战役", "淮海战役", "辽沈战役",
    "上甘岭战役", "嘉陵江战役", "松潘战役", "包座战役", "夏洮战役",
    "四渡赤水", "长征胜利", "井冈山会师", "懋功会师",
    "新中国成立", "抗战胜利", "红军长征",
    "上党战役", "挺进大别山", "五次战役",
    "重庆谈判", "北伐战争", "抗日战争",
    "解放战争", "人民解放战争", "抗日战争胜利",
    "抗美援朝", "抗美援朝战争",
    "游击战争", "革命战争", "太平洋战争", "土地革命战争", "民族革命战争",
    "苏维埃运动",
    "长征开始", "全面抗战爆发", "三大战役", "二次世界大战",
    # 以下为扩展事件（来自日记文本中常见提及）
    "反扫荡", "反摩擦", "反蚕食", "反顽斗争",
    "精兵简政", "三三制", "拥政爱民", "拥军优属",
    "整军运动", "练兵运动", "生产运动", "参军运动", "支前运动",
    "诉苦运动", "三查三整", "大整编",
    "减租减息运动", "查田运动", "清算斗争",
    "春季攻势", "夏季攻势", "秋季攻势", "冬季攻势",
    "济南战役", "石家庄战役", "临汾战役", "晋中战役",
    "襄樊战役", "睢杞战役", "开封战役", "兖州战役",
    "莱芜战役", "孟良崮战役", "鲁南战役", "宿北战役",
    "新开岭战役", "四平战役", "三下江南", "四保临江",
    "宜川战役", "西府战役", "荔北战役",
    "张家口战役", "集宁战役", "大同集宁战役",
    "绥远战役", "邯郸战役", "中原突围",
    "进军大别山", "豫东战役",
    "济南战役", "沈阳战役", "天津战役",
    "衡宝战役", "广西战役", "西南战役", "成都战役",
    "海南岛战役", "舟山群岛战役",
    "金城阻击战", "丁字山战役",
    "老秃山战斗", "马良山战斗",
    "延安保卫战", "陕北出击",
    "湘鄂川黔边战役", "陕甘宁边区保卫战",
    "劳山战役", "直罗镇战役", "东征战役", "西征战役",
    "山城堡战役", "宁夏战役", "河西战役",
    "神头岭战斗", "响堂铺战斗", "长乐村战斗", "香城固战斗",
    "齐会战斗", "陈庄战斗", "黄土岭战斗", "百团大战",
    "黄崖底战斗", "广阳战斗", "午城井沟战斗",
    "郓城战役", "曹县战役", "豫皖边战役",
    "盐城战役", "涟水战役", "两淮保卫战",
    "苏中战役", "七战七捷",
    "朝阳集战役", "定陶战役", "巨野战役", "鄄南战役",
    "滑县战役", "巨金鱼战役", "豫皖边战役",
    "平汉战役", "陇海路战役", "同蒲路战役",
    "易满战役", "保北战役", "清风店战役", "石家庄战役",
    "涞水战役", "察南绥东战役", "绥远战役",
    # 第三批补充：更多解放战争战役
    "青化砭战役", "羊马河战役", "蟠龙战役", "沙家店战役",
    "延清战役", "黄龙战役", "运城战役",
    "临汾战役", "汾孝战役",
    "宛西战役", "宛东战役", "豫东战役",
    "襄樊战役",
    "保南战役", "正太战役", "青沧战役", "大清河北战役",
    "石家庄战役", "晋中战役",
    "涟水保卫战", "盐城保卫战", "沭阳战役",
    "大别山反围攻", "高山铺战役",
    "确山战役", "洛阳战役",
    # 长征途中重要会议和事件
    "通道会议", "黎平会议", "猴场会议",
    "两河口会议", "毛儿盖会议", "巴西会议",
    "俄界会议", "哈达铺会议", "榜罗镇会议",
    "吴起镇会议", "下寺湾会议",
    "草地分兵", "南下北上",
    "飞夺泸定桥", "强渡大渡河", "巧渡金沙江",
    "腊子口战斗", "包座战斗",
    # 抗日战争重要战斗
    "夜袭阳明堡", "雁门关伏击", "七亘村伏击",
    "广阳伏击", "长生口战斗", "神头岭伏击",
    "响堂铺伏击", "长乐村急袭",
    "陆房突围", "大青山突围", "北岳反扫荡",
    "冀中五一反扫荡", "太行反扫荡", "太岳反扫荡",
    "沂蒙山反扫荡", "鲁中反扫荡", "胶东反扫荡",
    "潘溪渡战斗", "韩略村伏击",
    "甄家庄歼灭战", "田家会战斗",
    "宋庄战斗", "冉庄地道战",
    "百团大战",  # 已有一致
    # 根据地运动和建设
    "新式整军运动", "整党运动", "整训运动",
    "立功运动", "杀敌立功运动",
    "创立根据地", "开辟根据地", "巩固根据地",
    "武装保卫秋收", "坚壁清野",
    "民主建政", "普选运动",
    "变工互助", "合作化运动", "劳动互助",
    # 东北解放战争
    "秀水河子战斗", "鞍海战役",
    "新开岭战役", "四平保卫战", "四平攻坚战",
    "三下江南四保临江", "东北夏季攻势", "东北秋季攻势", "东北冬季攻势",
    # 其他
    "进军西藏", "昌都战役",
    "进军新疆",
    "大别山重建根据地", "洪湖革命根据地",
    "陕甘宁边区", "晋察冀边区", "晋冀鲁豫边区",
    "山东抗日根据地", "华中抗日根据地", "苏北抗日根据地",
    "淮南抗日根据地", "淮北抗日根据地",
    "大生产运动",  # 已有
    # 重要历史事件
    "二九运动", "一二一运动", "五二〇运动",
    "中美合作所",
    "军调处", "军事调处",
])

# 已知大事件的历史年份范围（仅在此年份内的提及才被提取）
KNOWN_EVENT_YEARS = {
    "北伐战争": (1926, 1928),
    "南昌起义": (1927, 1928),
    "秋收起义": (1927, 1928),
    "广州起义": (1927, 1928),
    "井冈山会师": (1928, 1929),
    "古田会议": (1929, 1930),
    "九一八事变": (1931, 1932),
    "湘江战役": (1934, 1935),
    "遵义会议": (1935, 1936),
    "红军长征": (1934, 1937),
    "长征开始": (1934, 1935),
    "长征胜利": (1936, 1937),
    "西安事变": (1936, 1937),
    "七七事变": (1937, 1938),
    "卢沟桥事变": (1937, 1938),
    "抗日战争": (1931, 1945),
    "抗日战争胜利": (1945, 1946),
    "抗战胜利": (1945, 1946),
    "全面抗战爆发": (1937, 1938),
    "平型关大捷": (1937, 1938),
    "台儿庄战役": (1938, 1939),
    "百团大战": (1940, 1942),
    "皖南事变": (1941, 1942),
    "整风运动": (1942, 1946),
    "大生产运动": (1942, 1946),
    "中共七大": (1945, 1946),
    "重庆谈判": (1945, 1946),
    "上党战役": (1945, 1946),
    "解放战争": (1945, 1950),
    "人民解放战争": (1945, 1950),
    "挺进大别山": (1947, 1948),
    "土地改革运动": (1947, 1950),
    "辽沈战役": (1948, 1949),
    "淮海战役": (1948, 1949),
    "平津战役": (1948, 1949),
    "三大战役": (1948, 1949),
    "渡江战役": (1949, 1950),
    "新中国成立": (1949, 1950),
    "抗美援朝": (1950, 1954),
    "抗美援朝战争": (1950, 1954),
    "五次战役": (1951, 1952),
    "上甘岭战役": (1952, 1953),
    "太平洋战争": (1941, 1946),
    "土地革命战争": (1927, 1937),
    "民族革命战争": (1931, 1945),
    "苏维埃运动": (1927, 1937),
    "湘南起义": (1928, 1929),
    "海陆丰起义": (1927, 1928),
    "渭华起义": (1928, 1929),
    "八七会议": (1927, 1928),
    "中共一大": (1921, 1922),
    "瓦窑堡会议": (1935, 1936),
    "洛川会议": (1937, 1938),
    "平型关战斗": (1937, 1938),
    "台儿庄大战": (1938, 1939),
    "二次世界大战": (1937, 1945),
    "革命战争": (1927, 1949),
    "游击战争": (1927, 1945),
    "五四运动": (1919, 1920),
    "新文化运动": (1915, 1926),
    "松潘战役": (1935, 1936),
    "包座战役": (1935, 1936),
    "夏洮战役": (1935, 1936),
    "嘉陵江战役": (1935, 1936),
    "四渡赤水": (1935, 1936),
    "懋功会师": (1935, 1936),
    # 扩展事件
    "反扫荡": (1939, 1945),
    "反摩擦": (1939, 1944),
    "反蚕食": (1941, 1944),
    "精兵简政": (1941, 1944),
    "三三制": (1940, 1946),
    "拥政爱民": (1943, 1949),
    "拥军优属": (1943, 1949),
    "整军运动": (1946, 1949),
    "练兵运动": (1945, 1949),
    "参军运动": (1945, 1949),
    "支前运动": (1946, 1949),
    "诉苦运动": (1947, 1949),
    "三查三整": (1947, 1949),
    "减租减息运动": (1937, 1947),
    "孟良崮战役": (1947, 1947),
    "鲁南战役": (1947, 1947),
    "济南战役": (1948, 1948),
    "石家庄战役": (1947, 1947),
    "宜川战役": (1948, 1948),
    "邯郸战役": (1945, 1945),
    "中原突围": (1946, 1946),
    "延安保卫战": (1947, 1947),
    "苏中战役": (1946, 1946),
    "定陶战役": (1946, 1946),
    "张家口战役": (1946, 1946),
    "绥远战役": (1945, 1945),
    "四平战役": (1946, 1947),
    "衡宝战役": (1949, 1949),
    "春季攻势": (1947, 1948),
    "夏季攻势": (1947, 1948),
    "秋季攻势": (1946, 1948),
    "冬季攻势": (1947, 1948),
    # 第三批年份约束
    "通道会议": (1934, 1934),
    "黎平会议": (1934, 1934),
    "猴场会议": (1934, 1935),
    "两河口会议": (1935, 1935),
    "毛儿盖会议": (1935, 1935),
    "巴西会议": (1935, 1935),
    "俄界会议": (1935, 1935),
    "哈达铺会议": (1935, 1935),
    "榜罗镇会议": (1935, 1935),
    "吴起镇会议": (1935, 1935),
    "草地分兵": (1935, 1936),
    "南下北上": (1935, 1936),
    "飞夺泸定桥": (1935, 1935),
    "强渡大渡河": (1935, 1935),
    "巧渡金沙江": (1935, 1935),
    "腊子口战斗": (1935, 1935),
    "夜袭阳明堡": (1937, 1937),
    "雁门关伏击": (1937, 1937),
    "七亘村伏击": (1937, 1937),
    "广阳伏击": (1937, 1937),
    "长生口战斗": (1938, 1938),
    "神头岭伏击": (1938, 1938),
    "响堂铺伏击": (1938, 1938),
    "长乐村急袭": (1938, 1938),
    "陆房突围": (1939, 1939),
    "大青山突围": (1941, 1942),
    "北岳反扫荡": (1941, 1944),
    "冀中五一反扫荡": (1942, 1942),
    "太行反扫荡": (1942, 1943),
    "太岳反扫荡": (1942, 1944),
    "沂蒙山反扫荡": (1941, 1942),
    "鲁中反扫荡": (1942, 1943),
    "胶东反扫荡": (1942, 1942),
    "潘溪渡战斗": (1941, 1941),
    "韩略村伏击": (1943, 1943),
    "甄家庄歼灭战": (1943, 1943),
    "田家会战斗": (1942, 1942),
    "宋庄战斗": (1942, 1942),
    "冉庄地道战": (1942, 1945),
    "新式整军运动": (1947, 1948),
    "整党运动": (1947, 1949),
    "立功运动": (1946, 1949),
    "变工互助": (1942, 1948),
    "合作化运动": (1943, 1949),
    "陕甘宁边区": (1937, 1950),
    "晋察冀边区": (1937, 1949),
    "晋冀鲁豫边区": (1937, 1949),
    "山东抗日根据地": (1937, 1945),
    "华中抗日根据地": (1938, 1945),
    "苏北抗日根据地": (1940, 1945),
    "淮南抗日根据地": (1940, 1945),
    "淮北抗日根据地": (1940, 1945),
    "一二九运动": (1935, 1936),
    "一二一运动": (1945, 1946),
    "青化砭战役": (1947, 1947),
    "羊马河战役": (1947, 1947),
    "蟠龙战役": (1947, 1947),
    "沙家店战役": (1947, 1947),
    "运城战役": (1947, 1948),
    "汾孝战役": (1947, 1947),
    "正太战役": (1947, 1947),
    "青沧战役": (1947, 1947),
    "保北战役": (1947, 1947),
    "洛阳战役": (1948, 1948),
    "宛西战役": (1948, 1948),
    "宛东战役": (1948, 1948),
    "睢杞战役": (1948, 1948),
    "豫东战役": (1948, 1948),
    "襄樊战役": (1948, 1948),
    "济南战役": (1948, 1948),
    "石家庄战役": (1947, 1947),
    "临汾战役": (1948, 1948),
    "晋中战役": (1948, 1948),
    "涟水保卫战": (1946, 1946),
    "盐城保卫战": (1946, 1946),
    "高山铺战役": (1947, 1947),
    "进军西藏": (1950, 1952),
    "昌都战役": (1950, 1951),
    "秀水河子战斗": (1946, 1946),
    "鞍海战役": (1946, 1946),
    "四平保卫战": (1946, 1946),
}

GENERIC_BLACKLIST = frozenset([
    "总结", "学习", "训练", "解决", "宣传", "动员", "部署",
    "传达", "讨论", "慰问", "救济", "分配", "征收",
    "生产", "开荒", "种地", "改善", "检查", "汇报",
    "参加", "组织", "举办", "召开", "成立", "建立",
    "进行", "举行", "开展", "贯彻", "实施", "实行",
    "通过", "制定", "颁布", "审查", "整理", "研究",
    "俘虏", "缴获", "歼灭", "击溃",
    "党员", "晚会", "演出", "比赛",
    "医院", "救治", "治病", "识字",
    "庆祝", "纪念", "工作", "会议", "运动", "活动",
    "整风", "土改",
    # 新增噪声词
    "讨论", "报告", "通知", "出发", "到达", "返回",
    "简单", "适当", "及时", "彻底",
])

GENERIC_PREFIXES = frozenset([
    "这", "那", "该", "各", "每", "某", "诸", "全", "本",
    "我", "你", "他", "她",
    "的", "了", "在", "有", "和", "与", "及", "或", "等",
    "对", "为", "由", "以", "被", "把", "将", "从",
    "一个", "一些", "很多", "大量", "许多", "部分",
    "这次", "那次", "每次", "首次", "再次",
    "这个", "那个", "这些", "那些",
    "进行", "开展", "举行", "参加", "组织",
])


def is_generic_name(name):
    """判断事件名是否为泛指/非重要"""
    if len(name) <= 3:
        return True
    for prefix in GENERIC_PREFIXES:
        if name.startswith(prefix):
            return True
    if re.match(r'^[大总小长新老高全](?:会议|战役|战斗|运动|工作|活动)$', name):
        return True
    if re.search(r'(?:我们|你们|他们|大家|同志|部队|敌军|我方|你们|自己|一个)', name):
        return True
    # 以"的"结尾且前面有长前缀
    if name.endswith('的') and len(name) >= 6:
        return True
    # 以时间段开头（"下午"、"上午"、"今天"、"本日"等）
    if re.match(r'^(?:今天|本日|今日|昨天|昨日|明天|明日|上午|下午|晚上|夜间|晨起|午后|晚间)', name):
        return True
    # 包含日程描述
    if re.search(r'(?:报告|通知|出发|到达|返回|进入|开始|结束|进行)', name):
        return True
    # 纯数字+后缀如 "一军团" 等保留, 但 "1个" 过滤
    if re.match(r'^\d+[个只件种次]', name):
        return True
    return False

# ============================================================
# 2. 正则模式
# ============================================================

NAMED_EVENT_PATTERN = re.compile(
    r'[^，。；：\s、！？]{2,8}(?:'
    r'战役|战斗|会战|大战|事变|起义|暴动|会师|'
    r'会议|运动|动员|竞赛|选举|纪念|庆祝|'
    r'大会|代表大会|条例|法令|'
    r'抗战|战争|攻势|反攻|胜利|大捷'
    r')'
)

ACTION_PATTERNS = {
    "军事作战": re.compile(
        r'(?:攻克|收复|解放|占领|攻占|夺取|歼灭|击溃|'
        r'围攻|突破|进驻|强渡|夜袭|奇袭|伏击|追击|'
        r'打通|粉碎|增援|截击|出击|围歼|扫荡|扫清|'
        r'摧毁|攻破|夺取|抗击|阻击|防守|死守|'
        r'突围|穿插|分割|包抄|合围)(?:了)?[^，。；\s]{2,8}'
    ),
    "组织建设": re.compile(
        r'(?:组织|开展|举办|召开|成立|建立|创办|发动|'
        r'推行|贯彻|实施|传达|讨论|总结|部署|'
        r'通过|制定|颁布|实行|改组|整顿|'
        r'纠正)(?:了)?[^，。；\s]{2,8}'
    ),
    "群众运动": re.compile(
        r'(?:慰问|宣传|援助|救济|捐献|征收|分配|'
        r'动员|号召|组织|发动|领导|带领|'
        r'减租|减息|土改|支前|劳军|拥军|'
        r'募集|募捐|救济)(?:了)?[^，。；\s]{2,8}'
    ),
    "政权建设": re.compile(
        r'(?:选举|投票|当选|任命|设立|划定|划分|'
        r'颁布|公布|施行|执行|建立|组建|'
        r'整编|改编|扩编|精简)(?:了)?[^，。；\s]{2,8}'
    ),
    "文化建设": re.compile(
        r'(?:出版|创办|编写|编辑|发行|印发|编印|油印|'
        r'演出|表演|演唱|排演|排练|公演|汇演|'
        r'宣传|张贴|刷写|绘制|创作|起草|'
        r'编报|编歌|编戏|排戏|演戏|教唱)(?:了)?[^，。；\s]{2,8}'
    ),
}

GENERIC_ACTION = re.compile(
    r'(?:进行|开展|举行|召开|组织|参加)(?:了)?[^，。；\s]{2,10}(?:'
    r'工作|活动|会议|运动|斗争|战斗|战役|竞赛|动员'
    r')'
)

PERSON_PATTERN = re.compile(r'[^，。；：\s、！？]{1,4}(?:同志|司令|政委|旅长|团长|'
                           r'师长|军长|主任|书记|部长|县长|队长|主席|'
                           r'总指挥|副总指挥|参谋长|政治部主任)')

LOCATION_PATTERN = re.compile(r'[^，。；：\s、！？]{2,6}(?:县|城|镇|村|庄|集|市|区|'
                             r'山|河|江|湖|关|口|桥|寨|岭|峰|'
                             r'省|府|郡|州|乡|里|堡|屯|营|庙)')

SIGNIFICANT_KW = {
    "军事作战": frozenset([
        "总攻", "决战", "大捷", "告捷", "凯旋", "胜利会师",
        "突围", "反攻", "全线出击", "扫荡", "围剿", "偷袭",
        "歼灭", "俘虏", "缴获", "击退", "击溃", "攻克",
        "渡江", "强渡", "夜袭", "奇袭", "伏击",
    ]),
    "组织建设": frozenset([
        "整风", "整改", "整党", "整军", "选举", "传达",
        "部署", "总结", "表彰", "批评与自我批评", "入党",
        "支部", "党员", "组织生活", "党性",
    ]),
    "群众运动": frozenset([
        "土改", "减租减息", "动员", "支前", "慰问", "宣传",
        "参军", "拥军", "劳军", "征粮", "募捐",
        "互助", "合作社", "变工", "换工",
    ]),
    "政权建设": frozenset([
        "选举", "参政", "建制", "改制", "划分", "新政",
        "法令", "条例", "税收", "征粮", "建政",
        "代表会议", "参议会", "三三制",
    ]),
    "日常生活": frozenset([
        "生产", "开荒", "种地", "丰收", "过节", "庆祝",
        "改善", "解决", "医院", "救治", "治病",
        "学习", "识字", "读报", "上课", "训练",
        "晚会", "演出", "比赛", "运动会",
    ]),
}
ALL_SIG_KW = frozenset().union(*SIGNIFICANT_KW.values())


def load_meta():
    with open(META_PATH, 'rb') as f:
        return pickle.load(f)


def load_sentiment():
    if not os.path.exists(SENTIMENT_PATH):
        return None
    with open(SENTIMENT_PATH) as f:
        return json.load(f)


def char_jaccard(a, b):
    if not a or not b:
        return 0
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union > 0 else 0


# ============================================================
# 3. 实体提取
# ============================================================

def extract_named_events(text, category, use_full_text=False, entry_year=None):
    """只精确匹配 KNOWN_EVENTS（大历史事件），不做正则提取"""
    events = []
    for known in KNOWN_EVENTS:
        if known in text:
            if entry_year is not None and known in KNOWN_EVENT_YEARS:
                y_min, y_max = KNOWN_EVENT_YEARS[known]
                if not (y_min <= entry_year <= y_max):
                    continue
            events.append({"name": known, "method": "known", "category": category})
    return events


def extract_action_events(text, category):
    events = []
    pattern = ACTION_PATTERNS.get(category)
    if pattern:
        for m in pattern.finditer(text):
            name = m.group().strip()
            if 3 <= len(name) <= 12 and name not in GENERIC_BLACKLIST and not is_generic_name(name):
                events.append({"name": name, "method": "action", "category": category})
    return events


def extract_keyword_events(text, category):
    events = []
    kw_set = SIGNIFICANT_KW.get(category, set()) | ALL_SIG_KW
    for kw in kw_set:
        if kw in GENERIC_BLACKLIST:
            continue
        if kw in text and text.count(kw) >= 2:
            events.append({"name": kw, "method": "keyword", "category": category})
    return events


def extract_persons(text):
    persons = set()
    for p in KNOWN_PERSONS:
        if p in text:
            persons.add(p)
    for m in PERSON_PATTERN.finditer(text):
        name = m.group().strip()
        if 2 <= len(name) <= 6 and name not in ("同志", "同志们", "我们"):
            for suffix in ["同志", "司令", "政委", "旅长", "团长", "师长", "军长",
                          "主任", "书记", "部长", "县长", "队长", "主席"]:
                if name.endswith(suffix) and len(name) > len(suffix):
                    name = name[:-len(suffix)]
                    break
            if 1 <= len(name) <= 4:
                persons.add(name)
    return persons


def extract_locations(text):
    locations = set()
    for loc in KNOWN_LOCATIONS:
        if loc in text:
            locations.add(loc)
    for m in LOCATION_PATTERN.finditer(text):
        loc = m.group().strip()
        if 3 <= len(loc) <= 8:
            locations.add(loc)
    return locations


def extract_organizations(text):
    orgs = set()
    for org in KNOWN_ORGANIZATIONS:
        if org in text:
            orgs.add(org)
    return orgs


def extract_entry_entities(entry, _=None):
    text = entry.get('text', '')
    category = entry.get('category', '') or ''
    entry_year = entry.get('year')

    result = {
        "named_events": [],
        "action_events": [],
        "keyword_events": [],
        "persons": set(), "locations": set(), "organizations": set(),
    }
    if len(text) < 15:
        return result

    # 大历史事件（KNOWN_EVENTS 精确匹配）
    result["named_events"] = extract_named_events(text, category, use_full_text=False, entry_year=entry_year)

    # 行动事件（加强质量过滤）
    action_events = extract_action_events(text, category)
    for ae in action_events:
        name = ae["name"]
        # 过滤条件
        if (5 <= len(name) <= 14
                and not any(p in name for p in '，。,．、；：！？""''（）【】《》?')
                and name not in GENERIC_BLACKLIST
                and not is_generic_name(name)
                and not any(name.startswith(p) for p in GENERIC_PREFIXES)
                and not any(name.startswith(p) for p in ('讨论', '报告', '通知', '总结', '部署', '传达', '汇报', '检查', '实行', '执行', '贯彻', '通过'))
                and not name.endswith(('的', '了', '在', '有', '和', '是', '的指示', '的材料', '的汇报', '的工作', '的会议', '的问题', '如下', '一段', '以后'))
                and not re.search(r'[""“」』;]', name)
                and not re.search(r'(?:积极性和自|半个地球|西北青年|托派组|四方面军第|无此秽|编队会议)', name)
                and not re.search(r'斗争(?:的基础|的结果|中|的|的)|的特殊|的认识|的问题|为\w{2,6}所', name)):
            result["action_events"].append(ae)

    # 关键词事件（仅当日记中同一关键词出现3次以上才保留）
    kw_set = ALL_SIG_KW
    text_lower = text
    for kw in kw_set:
        if kw in GENERIC_BLACKLIST:
            continue
        if kw in text_lower and text_lower.count(kw) >= 3:
            result["keyword_events"].append({"name": kw, "method": "keyword", "category": category})

    result["persons"] = extract_persons(text)
    result["locations"] = extract_locations(text)
    result["organizations"] = extract_organizations(text)
    return result


# ============================================================
# 4. 事件聚类
# ============================================================

# ============================================================
# 事件分类（六大类）
# ============================================================

CATEGORIES = ["军事作战", "组织建设", "群众运动", "政权建设", "文化建设", "日常生活"]

def classify_event(name):
    """根据事件名称自动归类到六大类之一"""
    # 文化建设（优先匹配短名以免误伤）
    if name in ("五四运动", "新文化运动", "一二九运动", "一二一运动", "五二〇运动"):
        return "文化建设"
    if any(kw in name for kw in ("宣传", "演出", "出版", "报刊", "创作",
                                  "戏剧", "歌咏", "墙报", "标语")):
        return "文化建设"
    # 政权建设
    if any(kw in name for kw in ("边区", "根据地", "民主建政", "普选",
                                  "苏维埃运动", "三三制", "新中国成立",
                                  "陕甘宁", "晋察冀", "晋冀鲁豫",
                                  "参议会", "代表会议", "法令",
                                  "重庆谈判")):
        return "政权建设"
    # 军事作战 — 专名后缀匹配
    if any(kw in name for kw in (
            "战役", "战斗", "会战", "大战", "事变", "起义", "暴动",
            "反扫荡", "突围", "保卫战", "攻坚战", "攻势", "反攻",
            "伏击", "大捷", "战争", "会师",
            "阻击战", "地道战", "破袭战",
            "歼灭战", "渡江战役")):
        return "军事作战"
    # 军事作战 — 专有操作名
    if any(kw in name for kw in (
            "夜袭", "奇袭", "强渡", "飞夺", "巧渡",
            "三下江南", "四保临江", "四渡赤水",
            "抗美援朝", "抗战胜利",
            "长征", "渡江", "进军")):
        return "军事作战"
    # 军事作战 — 行动动词
    if any(kw in name for kw in (
            "攻克", "收复", "解放", "占领", "攻占", "夺取",
            "围攻", "突破", "进驻", "追击",
            "打通", "粉碎", "增援", "出击", "围歼", "摧毁", "攻破",
            "抗击", "阻击", "死守", "穿插", "分割", "包抄",
            "歼灭", "击溃", "截击", "扫荡", "扫清",
            "反顽", "挺进")):
        return "军事作战"
    # 组织建设
    if any(kw in name for kw in ("会议", "整风", "整军", "整党",
                                  "精兵简政", "三查三整", "整训",
                                  "改组", "整顿", "调处",
                                  "传达", "部署", "贯彻")):
        return "组织建设"
    # 群众运动
    if any(kw in name for kw in ("减租", "土改", "拥政", "拥军", "参军",
                                  "支前", "诉苦", "立功", "变工",
                                  "合作化", "劳动互助", "募捐", "救济",
                                  "拥军优属", "拥政爱民", "劳军",
                                  "清算斗争")):
        return "群众运动"
    # 日常生活
    if any(kw in name for kw in ("生产运动", "大生产", "坚壁清野",
                                  "武装保卫", "开荒", "种地", "丰收",
                                  "识字", "读报", "晚会", "比赛",
                                  "运动会")):
        return "日常生活"
    # 以"运动"结尾的一般归群众运动
    if name.endswith("运动"):
        return "群众运动"
    # 以"战争"结尾的归军事
    if name.endswith("战争"):
        return "军事作战"
    # 行动事件按动词归类
    if any(kw in name for kw in (
            "组织", "开展", "举办", "召开", "成立", "建立",
            "创办", "发动", "推行", "实施",
            "讨论", "总结", "通过",
            "制定", "颁布", "实行")):
        return "组织建设"
    if any(kw in name for kw in (
            "慰问", "援助", "救济", "捐献",
            "征收", "分配", "号召", "领导", "带领")):
        return "群众运动"
    if any(kw in name for kw in (
            "出版", "编写", "编辑", "发行", "印发", "编印",
            "演出", "表演", "演唱", "排练", "公演",
            "绘制", "创作", "起草", "编报", "教唱")):
        return "文化建设"
    if any(kw in name for kw in (
            "选举", "投票", "任命", "设立", "划定",
            "颁布", "公布", "执行", "组建",
            "改编", "扩编", "精简")):
        return "政权建设"
    if any(kw in name for kw in (
            "医疗", "救治", "治病", "医院", "伙食",
            "住宿", "穿着", "文娱")):
        return "日常生活"
    return "其他"


def make_date_obj(entry):
    try:
        y, m, d = entry.get('year'), entry.get('month'), entry.get('day')
        if y and m and d:
            return datetime(int(y), int(m), int(d))
        if y:
            return datetime(int(y), 6, 15)
    except (ValueError, TypeError):
        pass
    return None


def cluster_events(all_extracted, meta):
    raw_events = []
    for entry_idx, (entry, extracted) in enumerate(zip(meta, all_extracted)):
        dt = make_date_obj(entry)
        if dt is None:
            continue
        diary_name = entry.get('diary_name', '')
        category = entry.get('category', '') or ''
        text = entry.get('text', '')[:120]

        seen = set()
        for ne in extracted["named_events"]:
            n = ne["name"]
            if n not in seen and 2 <= len(n) <= 12 and n not in GENERIC_BLACKLIST and not is_generic_name(n):
                seen.add(n)
                raw_events.append(dict(name=n, date=dt, diary=diary_name, entry_idx=entry_idx,
                                       category=category, type="named", text_snippet=text))

        # 行动事件
        for ae in extracted.get("action_events", []):
            n = ae["name"]
            if n not in seen and 4 <= len(n) <= 14 and n not in GENERIC_BLACKLIST and not is_generic_name(n):
                seen.add(n)
                raw_events.append(dict(name=n, date=dt, diary=diary_name, entry_idx=entry_idx,
                                       category=category, type="action", text_snippet=text))

        # 关键词事件
        for ke in extracted.get("keyword_events", []):
            n = ke["name"]
            if n not in seen and 2 <= len(n) <= 8 and n not in GENERIC_BLACKLIST and not is_generic_name(n):
                seen.add(n)
                raw_events.append(dict(name=n, date=dt, diary=diary_name, entry_idx=entry_idx,
                                       category=category, type="keyword", text_snippet=text))

    raw_events.sort(key=lambda x: x["date"])

    def name_key(name):
        # 1) 如果事件名以某个已知重大事件结尾，归并到该已知事件
        for known in sorted(KNOWN_EVENTS, key=len, reverse=True):
            if name.endswith(known):
                return known
        # 2) 一般后缀归一化：反复截短前缀、剥掉左右两侧虚词
        for suf in sorted(['战役', '战斗', '会战', '大战', '事变', '起义', '暴动', '会师',
                           '会议', '运动', '动员', '选举', '纪念', '庆祝',
                           '抗战', '战争', '攻势', '反攻', '胜利', '大捷', '谈判',
                           '大会', '代表大会', '条例', '法令'], key=len, reverse=True):
            if name.endswith(suf) and len(name) > len(suf):
                prefix = name[:-len(suf)]
                # 反复截短 + 剥左右虚词，直到稳定
                fillers = frozenset('的了在上下中前后内外以')
                changed = True
                while changed:
                    changed = False
                    if len(prefix) > 4:
                        prefix = prefix[-4:]
                        changed = True
                    while len(prefix) > 2 and prefix[0] in fillers:
                        prefix = prefix[1:]
                        changed = True
                    while len(prefix) > 2 and prefix[-1] in fillers:
                        prefix = prefix[:-1]
                        changed = True
                if not prefix:
                    return suf
                candidate = prefix + suf
                if candidate != name:
                    return name_key(candidate)
                return candidate
        return name

    # 按规范化名称分组（同时对映射到年份受限事件的条目做二次校验）
    name_groups = defaultdict(list)
    for e in raw_events:
        nk = name_key(e["name"])
        if nk in KNOWN_EVENT_YEARS:
            y_min, y_max = KNOWN_EVENT_YEARS[nk]
            ey = e["date"].year
            if ey and not (y_min <= ey <= y_max):
                continue
        name_groups[nk].append(e)

    # 组内按日期合并（同名事件全部合并到同一簇，不设时间窗口）
    merged = []
    for nk, group in name_groups.items():
        group.sort(key=lambda x: x["date"])
        cur = dict(name=nk, date=group[0]["date"],
                   start_date=group[0]["date"], end_date=group[0]["date"],
                   diaries={group[0]["diary"]}, members=[group[0]])
        for e in group[1:]:
            cur["members"].append(e)
            cur["end_date"] = max(cur["end_date"], e["date"])
            cur["diaries"].add(e["diary"])
        merged.append(cur)

    merged.sort(key=lambda x: x["start_date"])

    # 第二遍：将 action/keyword 事件簇与附近（±45天）的已知事件簇合并
    known_clusters = [cl for cl in merged if any(m.get("type") == "named" for m in cl["members"])]
    other_clusters = [cl for cl in merged if cl not in known_clusters]
    for ocl in other_clusters:
        merged_with_known = False
        for kcl in known_clusters:
            gap = min(
                abs((ocl["start_date"] - kcl["end_date"]).days),
                abs((kcl["start_date"] - ocl["end_date"]).days),
                (ocl["start_date"] - kcl["start_date"]).days if ocl["start_date"] > kcl["start_date"]
                else (kcl["start_date"] - ocl["start_date"]).days,
            )
            if gap <= 45:
                kcl["members"].extend(ocl["members"])
                kcl["start_date"] = min(kcl["start_date"], ocl["start_date"])
                kcl["end_date"] = max(kcl["end_date"], ocl["end_date"])
                kcl["diaries"].update(ocl["diaries"])
                merged_with_known = True
                break
        if not merged_with_known:
            known_clusters.append(ocl)
    merged = known_clusters
    merged.sort(key=lambda x: x["start_date"])
    return merged


# ============================================================
# 5. 重要性评分
# ============================================================

def compute_importance(cluster, total_entries):
    n_mentions = len(cluster["members"])
    n_diaries = len(cluster["diaries"])
    days_span = (cluster["end_date"] - cluster["start_date"]).days + 1

    name = cluster["name"]
    is_named = any(name.endswith(suf) for suf in
                   ['战役', '战斗', '会战', '大战', '事变', '起义', '暴动', '会师',
                    '会议', '运动', '动员', '选举', '纪念', '庆祝',
                    '抗战', '战争', '攻势', '反攻', '胜利', '大捷', '谈判'])
    is_short = len(name) <= 3 and not is_named

    score = (
        0.30 * math.log1p(n_mentions) / math.log1p(100) +
        0.35 * min(n_diaries / 10, 1.0) +
        0.15 * min(days_span / 90, 1.0) +
        0.20 * (1.0 if is_named else 0.5 if is_short else 0.7)
    )
    return round(min(score, 1.0), 4)


# ============================================================
# 6. 实体统计
# ============================================================

def build_entity_registry(all_extracted, meta):
    person_counts = defaultdict(lambda: {"count": 0, "diaries": set()})
    loc_counts = defaultdict(lambda: {"count": 0, "diaries": set()})
    org_counts = defaultdict(lambda: {"count": 0, "diaries": set()})
    person_years = defaultdict(lambda: {"min": 9999, "max": 0})

    for entry, extracted in zip(meta, all_extracted):
        diary = entry.get('diary_name', '')
        year = entry.get('year') or 0
        for p in extracted["persons"]:
            person_counts[p]["count"] += 1
            person_counts[p]["diaries"].add(diary)
            if year:
                person_years[p]["min"] = min(person_years[p]["min"], year)
                person_years[p]["max"] = max(person_years[p]["max"], year)
        for loc in extracted["locations"]:
            loc_counts[loc]["count"] += 1
            loc_counts[loc]["diaries"].add(diary)
        for org in extracted["organizations"]:
            org_counts[org]["count"] += 1
            org_counts[org]["diaries"].add(diary)

    def build(counter, years=None):
        result = []
        for name, data in sorted(counter.items(), key=lambda x: -x[1]["count"]):
            entry = {"name": name, "mention_count": data["count"], "diary_count": len(data["diaries"])}
            if years and name in years:
                entry["first_year"] = years[name]["min"]
                entry["last_year"] = years[name]["max"]
            result.append(entry)
        return result

    return {
        "persons": build(person_counts, person_years),
        "locations": build(loc_counts),
        "organizations": build(org_counts),
    }


def compute_sentiment_impact(cluster, date_index):
    if date_index is None:
        return None
    event_date = cluster["start_date"]
    window = 30
    before, after = [], []
    for d_off in range(-window, window + 1):
        if d_off == 0:
            continue
        d = event_date + timedelta(days=d_off)
        scores = date_index.get(d.strftime("%Y-%m-%d"), [])
        if d_off < 0:
            before.extend(scores)
        else:
            after.extend(scores)
    if len(before) < 3 and len(after) < 3:
        return None
    def avg(arr):
        return round(sum(arr) / len(arr), 4) if arr else None
    return {
        "before_mean": avg(before), "after_mean": avg(after),
        "change": round(avg(after) - avg(before), 4) if before and after else None,
        "sample_before": len(before), "sample_after": len(after),
    }


def build_date_index(meta, sentiment_data):
    if not sentiment_data:
        return None
    entries = sentiment_data.get('entries', [])
    idx = defaultdict(list)
    for e, s in zip(meta, entries):
        dt = make_date_obj(e)
        if dt is None:
            continue
        idx[dt.strftime("%Y-%m-%d")].append(s.get('score', 0))
    return dict(idx)


# ============================================================
# 7. 主流程
# ============================================================

def extract_events_main():
    print("=" * 50)
    print("📅 革命日记事件脉络提取")
    print("=" * 50)

    print("\n📚 加载条目元数据...")
    meta = load_meta()
    print(f"   共 {len(meta)} 条日记")

    print("\n📖 加载情感分析结果...")
    sentiment_data = load_sentiment()
    print(f"   已加载 {sentiment_data['total']} 条情感数据" if sentiment_data else "   未找到")

    total = len(meta)
    print("\n🔍 逐条提取实体...")
    all_extracted = []
    for i, entry in enumerate(meta):
        all_extracted.append(extract_entry_entities(entry))
        if (i + 1) % 2000 == 0:
            print(f"   进度: {i+1}/{total} ({(i+1)/total*100:.0f}%)")
    print(f"   完成: {total}/{total}")

    total_events = sum(len(e["named_events"]) for e in all_extracted)
    total_persons = sum(len(e["persons"]) for e in all_extracted)
    print(f"\n   提取: {total_events} 事件, {total_persons} 人物提及")

    print("\n🔗 事件聚类...")
    clusters = cluster_events(all_extracted, meta)
    print(f"   聚类后: {len(clusters)} 个事件簇")
    meaningful = [c for c in clusters if len(c["members"]) >= 2 or len(c["diaries"]) >= 2]
    print(f"   有意义: {len(meaningful)} 个")

    print("⭐ 计算重要性...")
    date_index = build_date_index(meta, sentiment_data) if sentiment_data else None
    meaningful_ids = {id(c) for c in meaningful}

    for i, cl in enumerate(clusters):
        cl["importance"] = compute_importance(cl, total)
        cl_persons, cl_locs, cl_orgs = set(), set(), set()
        for m in cl["members"]:
            idx = m["entry_idx"]
            if idx < len(all_extracted):
                cl_persons.update(all_extracted[idx]["persons"])
                cl_locs.update(all_extracted[idx]["locations"])
                cl_orgs.update(all_extracted[idx]["organizations"])
        cl["related_persons"] = sorted(cl_persons)[:10]
        cl["related_locations"] = sorted(cl_locs)[:10]
        cl["related_orgs"] = sorted(cl_orgs)[:5]
        if id(cl) in meaningful_ids:
            cl["sentiment_impact"] = compute_sentiment_impact(cl, date_index)
        if (i + 1) % 3000 == 0:
            print(f"   进度: {i+1}/{len(clusters)}")

    clusters.sort(key=lambda x: -x["importance"])

    print("🏛️ 构建实体注册表...")
    entity_registry = build_entity_registry(all_extracted, meta)
    print(f"   人物: {len(entity_registry['persons'])}")
    print(f"   地点: {len(entity_registry['locations'])}")
    print(f"   组织: {len(entity_registry['organizations'])}")

    print("💾 组装输出（质量过滤）...")
    output_clusters = []
    skipped_generic = 0
    skipped_low_imp = 0
    skipped_rare = 0
    for cl in clusters:
        # 质量门槛
        name = cl["name"]
        if cl["importance"] < 0.15:
            skipped_low_imp += 1
            continue
        if is_generic_name(name):
            skipped_generic += 1
            continue
        types = {m["type"] for m in cl["members"]}
        if types <= {"action", "keyword"} and len(cl["members"]) < 3:
            skipped_rare += 1
            continue

        mentions = []
        seen_mention = set()
        for m in cl["members"]:
            key = (m["diary"], m["date"].strftime("%Y-%m-%d"), m["name"])
            if key not in seen_mention:
                seen_mention.add(key)
                mentions.append(dict(
                    diary_name=m["diary"],
                    date=m["date"].strftime("%Y-%m-%d"),
                    text_snippet=m["text_snippet"][:120],
                    entry_idx=m["entry_idx"],
                ))
        mentions.sort(key=lambda x: x["date"])
        if len(mentions) > 30:
            mentions = mentions[:30]
        output_clusters.append(dict(
            id=f"evt_{len(output_clusters)+1:04d}",
            name=cl["name"],
            category=classify_event(cl["name"]),
            start_date=cl["start_date"].strftime("%Y-%m-%d"),
            end_date=cl["end_date"].strftime("%Y-%m-%d"),
            importance=cl["importance"],
            total_mentions=len(cl["members"]),
            diary_count=len(cl["diaries"]),
            diaries=sorted(cl["diaries"]),
            mentions=mentions,
            related_entities=dict(
                persons=cl.get("related_persons", []),
                locations=cl.get("related_locations", []),
                organizations=cl.get("related_orgs", []),
            ),
            sentiment_impact=cl.get("sentiment_impact"),
        ))

    cross_diary = [c for c in output_clusters if c["diary_count"] >= 2]
    stats = dict(
        total_event_clusters=len(output_clusters),
        total_entity_persons=len(entity_registry["persons"]),
        total_entity_locations=len(entity_registry["locations"]),
        total_entity_organizations=len(entity_registry["organizations"]),
        cross_diary_event_count=len(cross_diary),
        cross_diary_event_pct=round(len(cross_diary)/max(len(output_clusters),1)*100, 1),
    )

    output = dict(
        version="1.0", total_entries=total, extraction_method="regex_dictionary",
        generated_at=datetime.now().isoformat(),
        statistics=stats, event_clusters=output_clusters, entities=entity_registry,
    )

    print(f"\n📊 统计: {stats['total_event_clusters']} 事件簇, "
          f"{stats['cross_diary_event_count']} 跨日记 ({stats['cross_diary_event_pct']}%)")
    if skipped_generic:
        print(f"   过滤泛指: {skipped_generic}, 低重要性: {skipped_low_imp}, 低频: {skipped_rare}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    sz = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"💾 保存到 {OUTPUT_PATH} ({sz:.2f}MB)")
    print("✅ 完成！")
    return output


if __name__ == "__main__":
    extract_events_main()
