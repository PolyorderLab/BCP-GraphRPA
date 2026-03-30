using Pkg; Pkg.activate("/home/zxy/project/jl") 
using Random
using YAML
using Polymer
using PolymerArchitecture
using PolymerArchitectureMakie: visualize
using CairoMakie  # 静态图片渲染后端

# =================================================================
# 工具函数：从 YML 文件逆向重构 BlockCopolymer 对象
# =================================================================
function load_bcp_from_yml(filepath)
    # 读取 YML 文件
    data = YAML.load_file(filepath)
    blocks_data = data["system"]["components"][1]["blocks"]
    
    # 提取拓扑类型，如果没有则默认为 Random
    topo_str = get(data["system"], "topology", "Random")
    topo_type = Symbol(topo_str)

    # 1. 统计每个节点的连接度 (degree)，以区分 BranchPoint 和 FreeEnd
    node_degrees = Dict{String, Int}()
    for b in blocks_data
        for node_name in b["ends"]
            node_degrees[node_name] = get(node_degrees, node_name, 0) + 1
        end
    end

    # 2. 实例化节点对象
    node_objs = Dict{String, Any}()
    for (name, deg) in node_degrees
        sym_name = Symbol(name)
        node_objs[name] = deg > 1 ? BranchPoint(sym_name) : FreeEnd(sym_name)
    end

    # 3. 实例化嵌段对象
    # 【修复核心就在这一行：把 blocks = [] 改为 blocks = PolymerBlock[]】
    blocks = PolymerBlock[] 
    
    for b in blocks_data
        label = Symbol(b["segment"])
        seg = KuhnSegment(Symbol(b["segment"]))
        f = Float64(b["length"])
        n1 = node_objs[b["ends"][1]]
        n2 = node_objs[b["ends"][2]]
        
        push!(blocks, PolymerBlock(label, seg, f, n1, n2))
    end

    # 4. 组装并返回聚合物对象
    return BlockCopolymer(topo_type, blocks; check=false)
end

# =================================================================
# 主程序：随机抽取并画图
# =================================================================
function main()
    data_dir = "/home/zxy/project/TAPD/TrainData/data/train_topo"
    vis_dir = "/home/zxy/project/TAPD/TrainData/data/topo_visualizations"
    mkpath(vis_dir) # 创建专门存放图片的文件夹

    # 找到所有的 yml 文件
    all_files = filter(f -> endswith(f, ".yml"), readdir(data_dir))
    if isempty(all_files)
        println("❌ 在 $data_dir 目录下没有找到任何 .yml 文件！")
        return
    end

    # 随机抽取 10 个文件 (如果总数不足10个则全取)
    sample_size = min(100, length(all_files))
    selected_files = shuffle(all_files)[1:sample_size]

    println("🎨 开始渲染聚合物拓扑结构图...")
    
    for (i, file) in enumerate(selected_files)
        filepath = joinpath(data_dir, file)
        
        # 解析高分子对象
        bcp = load_bcp_from_yml(filepath)
        
        # 使用 Makie 生成图像
        # PolymerArchitectureMakie 的 visualize() 会返回一个 Figure 对象
        fig = visualize(bcp)
        
        # 提取文件前缀作为图片名的一部分 (比如 sample_00012)
        base_name = replace(file, ".yml" => "")
        topo_name = string(bcp.label)
        
        # 设定保存路径
        save_path = joinpath(vis_dir, "$(base_name)_$(topo_name).png")
        
        # 保存为 PNG 图片 (使用 CairoMakie)
        save(save_path, fig)
        
        println(" [$(i)/$(sample_size)] 保存成功: $(base_name)_$(topo_name).png")
    end
    
    println("\n✅ 绘图任务完成！所有图片已保存在: $vis_dir")
end

# 运行主程序
main()