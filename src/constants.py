# 定义拓扑类型到数字分类标签的映射字典
TOPO_MAP = {
    "Random_2blocks": 0,
    "Random_3blocks": 1,
    "Random_4blocks": 2,
    "Random_5blocks": 3,
    "Random_6blocks": 4
}

TOPO_MAP_REV = {v: k for k, v in TOPO_MAP.items()}

COLOR_MAP_TOPO = {
    0: 'green',       # Random_2blocks
    1: 'royalblue',   # Random_3blocks
    2: 'magenta',     # Random_4blocks
    3: 'orange',      # Random_5blocks
    4: 'purple'       # Random_6blocks
}
