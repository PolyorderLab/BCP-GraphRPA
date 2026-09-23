using Pkg; Pkg.activate(@__DIR__)
using Roots
using Optim
using Polymer
using PolymerArchitecture
include(joinpath(@__DIR__, "rpa", "ExtractedPolyorderRPA.jl"))
using .ExtractedPolyorderRPA: RPA
using Random, Statistics, Graphs
using LinearAlgebra
using YAML
using MAT

BLAS.set_num_threads(1)

# ================= 1. 生成链结构的唯一签名(用于去重) =================
function get_chain_signature(chain::BlockCopolymer, topo_type::String)
    blocks = chain.blocks
    n = length(blocks)
    
    # 提取 segment 类型序列 (A=1, B=0)
    seg_types = Int[]
    for b in blocks
        label = string(b.segment.label)
        push!(seg_types, occursin("A", label) ? 1 : 0)
    end
    
    # 嵌段数量不同,精度不同:
    # - 2嵌段:使用 6 位小数(确保能生成 30000+ 个唯一签名)
    # - 3-5嵌段:使用 4 位小数
    if n == 2
        f_vals = round.([b.f for b in blocks], digits=6)
    else
        f_vals = round.([b.f for b in blocks], digits=4)
    end
    
    # 构建签名:拓扑类型 + block数 + segment序列 + f序列
    seg_str = join(seg_types, ",")
    f_str   = join(f_vals, ",")
    
    return "$(topo_type)_$(n)_$(seg_str)_$(f_str)"
end

# ================= 2. 高效且不越界的随机体积分数切分函数 =================
function rand_positive_numbers(rng::AbstractRNG, n::Int, total::Float64, minimum::Float64)
    total_int = round(Int, 1000 * total)
    min_int = round(Int, 1000 * minimum)
    numbers = [0]
    for i in 1:n-1
        upper_bound = total_int - (n-i)*min_int - sum(numbers)
        if min_int <= upper_bound
            number = rand(rng, min_int:upper_bound)
        else
            number = min_int # 兜底限幅
        end
        push!(numbers, number)
    end
    push!(numbers, total_int - sum(numbers))
    return numbers[2:end] ./ 1000.0
end

# ================= 新增:纯随机图论生长算法核心 =================
# 1. 高效且不越界的随机体积分数切分函数
function get_random_f(rng::AbstractRNG, num_segments::Int, total_f::Float64, f_min::Float64)
    if num_segments == 0 return Float64[] end
    if num_segments == 1 return [total_f] end
    if f_min * num_segments >= total_f
        return fill(total_f / num_segments, num_segments)
    end
    remaining = total_f - (num_segments * f_min)
    temp_pts = sort(rand(rng, num_segments - 1))
    gaps = [temp_pts[1]; diff(temp_pts); 1.0 - temp_pts[end]]
    return gaps .* remaining .+ f_min
end

# 2. 完美无环、智能着色的随机拓扑生成器
function generate_random_architecture(rng::AbstractRNG, total_blocks::Int; minimum_f=0.03)
    # [步骤一] 随机整体体积分数
    # 两嵌段使用更宽的范围 [0.001, 0.999],提供 ~1000*998 ≈ 998000 个唯一 fA 值
    # 配合 6 位小数的签名精度,理论上可生成 ~10^6 级别唯一样本
    if total_blocks == 2
        fA = rand(rng) * 0.998 + 0.001  # 连续均匀采样 [0.001, 0.999]
    else
        fA = rand(rng) * 0.90 + 0.05  # 3-5嵌段保持 [0.05, 0.95]
    end
    
    # [步骤二] 树状图(Tree)严格生长,数学上100%无环
    edges = Tuple{Int, Int}[]
    degrees = Dict{Int, Int}(1 => 0)
    active_nodes = [1]
    
    for new_node in 2:(total_blocks + 1)
        # 为了符合高分子物理,限制单个节点最多连 4 根线
        candidates = [n for n in active_nodes if degrees[n] < 4]
        if isempty(candidates)
            candidates = active_nodes
        end
        parent = rand(rng, candidates)
        push!(edges, (parent, new_node))
        
        degrees[parent] += 1
        degrees[new_node] = 1
        push!(active_nodes, new_node)
    end
    
    # [步骤三] 图着色:强制过滤无效的"同色线型相连"
    edge_colors = Vector{Symbol}(undef, total_blocks)
    edge_adj = [Int[] for _ in 1:total_blocks]
    # 找到所有共享度数为 2 节点的边 (即线型相连的边)
    for i in 1:total_blocks
        for j in i+1:total_blocks
            u1, v1 = edges[i]
            u2, v2 = edges[j]
            shared = intersect([u1, v1], [u2, v2])
            if !isempty(shared) && degrees[shared[1]] == 2
                push!(edge_adj[i], j)
                push!(edge_adj[j], i)
            end
        end
    end
    
    # 贪心着色:相邻线型边必须异色
    for i in 1:total_blocks
        forbidden = [edge_colors[j] for j in edge_adj[i] if isassigned(edge_colors, j)]
        choices = setdiff([:A, :B], forbidden)
        # 如果是复杂星型节点导致没得选,才随机分配;线型必定异色
        edge_colors[i] = isempty(choices) ? rand(rng, [:A, :B]) : rand(rng, choices)
    end
    
    # 确保体系必须是共聚物(有A也有B)
    if !(:A in edge_colors) edge_colors[rand(rng, 1:total_blocks)] = :A end
    if !(:B in edge_colors) edge_colors[rand(rng, 1:total_blocks)] = :B end
    
    # [步骤四] 分配体积分数
    num_A = count(==( :A ), edge_colors)
    num_B = total_blocks - num_A
    f_A_list = get_random_f(rng, num_A, fA, minimum_f)
    f_B_list = get_random_f(rng, num_B, 1.0 - fA, minimum_f)
    
    # [步骤五] 打包实体 PolymerBlock
    idx_A, idx_B = 1, 1
    blocks = PolymerBlock[]
    
    for i in 1:total_blocks
        u, v = edges[i]
        # 根据统计好的度数,精准判断是 FreeEnd 还是 BranchPoint
        node_u = degrees[u] > 1 ? BranchPoint(Symbol("N$u")) : FreeEnd(Symbol("N$u"))
        node_v = degrees[v] > 1 ? BranchPoint(Symbol("N$v")) : FreeEnd(Symbol("N$v"))
        
        color = edge_colors[i]
        if color == :A
            f_val = f_A_list[idx_A]
            idx_A += 1
        else
            f_val = f_B_list[idx_B]
            idx_B += 1
        end
        
        b = PolymerBlock(Symbol("b$i"), KuhnSegment(color), f_val, node_u, node_v)
        push!(blocks, b)
    end
    
    chain = BlockCopolymer(Symbol("AB"), blocks; check=false)
    topo_type = "Random_$(total_blocks)blocks"
    
    return chain, topo_type, total_blocks
end

# ================= 原有:向GNN特征矩阵的提取映射 (保持不变) =================
function polymer_to_gnn(chain; N_max::Int=20, D_feat::Int=2)
    blocks = chain.blocks
    N_blocks = length(blocks)
    graph = BlockCopolymerGraph(chain)
    Nv = Graphs.nv(graph)
    
    if Nv > N_max 
        return nothing 
    end
    
    degs = Graphs.degree(graph)
    F_pad = zeros(Float32, N_max, D_feat)
    
    for i in 1:Nv
        F_pad[i, 1] = Float32(degs[i])
        neighbor_nodes = Graphs.neighbors(graph, i)
        if !isempty(neighbor_nodes)
            avg_neighbor_deg = sum(degs[n] for n in neighbor_nodes) / length(neighbor_nodes)
            F_pad[i, 2] = Float32(avg_neighbor_deg)
        else
            F_pad[i, 2] = 0.0f0
        end
    end
    
    edges_obj = collect(Graphs.edges(graph))
    num_edges = length(edges_obj)
    
    edge_index = zeros(Int64, 2, num_edges * 2)
    edge_attr = zeros(Float32, num_edges * 2, 2)
    
    for (idx, e) in enumerate(edges_obj)
        u, v = Graphs.src(e), Graphs.dst(e)
        
        # 使用 Polymer._sort_tuple2 来匹配 edge2block 的键格式
        edge_key = u < v ? (u, v) : (v, u)

        if haskey(graph.edge2block, edge_key)
            block = graph.edge2block[edge_key]
            f_val = Float32(block.f)
            label_str = string(block.segment.label)
            type_val = occursin("A", label_str) ? 1.0f0 : 0.0f0
        else
            @warn "未找到边 ($u, $v) 对应的块属性,使用默认值"
            f_val, type_val = 0.0f0, 0.5f0
        end
        
        edge_index[1, 2idx-1], edge_index[2, 2idx-1] = u, v
        edge_attr[2idx-1, 1], edge_attr[2idx-1, 2] = f_val, type_val
        
        edge_index[1, 2idx], edge_index[2, 2idx] = v, u
        edge_attr[2idx, 1], edge_attr[2idx, 2] = f_val, type_val
    end
    
    return F_pad, edge_index, edge_attr
end

# ================= 多线程并行采样与 RPA 计算 (按嵌段数量分组生成) =================
function run_generation_by_blockcount(targets::Dict{Int, Int})
    # targets: Dict{嵌段数 => 目标数量}
    
    datasets = Dict{Int, Any}()
    seen_signatures = Dict{Int, Set{String}}()
    data_locks = Dict{Int, ReentrantLock}()
    stats = Dict{Int, Dict{String, Int}}()
    
    for n in keys(targets)
        datasets[n] = []
        seen_signatures[n] = Set{String}()
        data_locks[n] = ReentrantLock()
        stats[n] = Dict{String, Int}()
    end
    
    seen_lock = ReentrantLock()
    
    num_threads = Threads.nthreads()
    total_target = sum(values(targets))
    target_keys = collect(keys(targets))
    println("多线程分组生产模式 | 线程数: $num_threads | 目标总量: $total_target")
    for n in sort(target_keys)
        println("  - $n 嵌段: $(targets[n]) 个")
    end
    start_time = time()

    @sync for t in 1:num_threads
        Threads.@spawn begin
            local_rng = MersenneTwister(generation_seed + Threads.threadid())
            
            while true
                # 检查是否所有组都已达标
                all_done = true
                for n in sort(target_keys)
                    c = lock(data_locks[n]) do
                        length(datasets[n]) < targets[n]
                    end
                    if c
                        all_done = false
                        break
                    end
                end
                all_done && break
                
                # 随机选择一个还未达标的嵌段数量
                candidates = [n for n in target_keys if lock(data_locks[n]) do
                    length(datasets[n]) < targets[n]
                end]
                isempty(candidates) && break
                target_blocks = rand(local_rng, candidates)
                
                chain, topo_type, n_b = generate_random_architecture(local_rng, target_blocks; minimum_f=0.03)
                n_max_val = 32
                
                # 去重检查
                sig = get_chain_signature(chain, topo_type)
                is_new = lock(seen_lock) do
                    if sig in seen_signatures[target_blocks]
                        false
                    else
                        push!(seen_signatures[target_blocks], sig)
                        true
                    end
                end
                is_new || continue
                
                try
                    res = RPA.compute_stability_limit(chain; χN0=15.0, k2_max=100.0)
                    χN_star = res[1]
                    k_star = res[2]
                    
                    if 0.1 < χN_star < 5000.0
                        gnn_out = polymer_to_gnn(chain; N_max=n_max_val, D_feat=3)
                        
                        if !isnothing(gnn_out)
                            lock(data_locks[target_blocks]) do
                                if length(datasets[target_blocks]) < targets[target_blocks]
                                    push!(datasets[target_blocks], (gnn_out..., Float32(χN_star), Float32(k_star), topo_type, chain))
                                    stats[target_blocks][topo_type] = get(stats[target_blocks], topo_type, 0) + 1
                                    
                                    # 统计总体进度
                                    current_counts = [length(datasets[n]) for n in sort(target_keys)]
                                    total_current = sum(current_counts)
                                    
                                    if total_current % 500 == 0
                                        elapsed = round(time() - start_time, digits=1)
                                        percent = round(total_current / total_target * 100, digits=2)
                                        sps = total_current / elapsed
                                        eta_h = floor(Int, (total_target - total_current) / sps / 3600)
                                        eta_m = floor(Int, ((total_target - total_current) / sps % 3600) / 60)
                                        
                                        count_str = join(["$n:$v" for (n,v) in zip(sort(target_keys), current_counts)], " ")
                                        dup_total = sum(length, values(seen_signatures))
                                        dup_pct = round(dup_total / (total_current + dup_total) * 100, digits=1)
                                        print("\r\x1b[K>> $percent% ($total_current/$total_target) | 过滤:$dup_pct% | ETA: $(eta_h)h$(eta_m)m | $count_str")
                                        flush(stdout)
                                    end
                                end
                            end
                        end
                    end
                catch
                    continue
                end
            end
        end
    end
    
    println("\n\n 生成完成!各嵌段数量分布:")
    stat_keys = collect(keys(stats))
    for n in sort(stat_keys)
        println("  [$n 嵌段]")
        for (k, v) in stats[n]
            println("    - $k: $v 样本")
        end
    end
    return datasets, stats
end

function option(name, default)
    i = findfirst(==(name), ARGS)
    isnothing(i) && return default
    i < length(ARGS) || error("Missing value for $name")
    return ARGS[i + 1]
end
generation_seed = parse(Int, option("--seed", "42"))
count_per_block = parse(Int, option("--count-per-block", "30000"))
count_per_block > 0 || error("Count must be positive")
output_base_dir = option("--output-dir", joinpath(@__DIR__, "..", "datasets", "raw_2to5"))
ispath(output_base_dir) && error("Output already exists: $output_base_dir")
mkpath(output_base_dir)

# 定义每个嵌段数量的目标数量
targets = Dict(n => count_per_block for n in 2:5)
datasets, final_stats = run_generation_by_blockcount(targets)

txt_path = joinpath(output_base_dir, "nnumber_summary.txt")
open(txt_path, "w") do file
    println(file, "生成完成!各嵌段数量分布:")
    final_stat_keys = collect(keys(final_stats))
    for n in sort(final_stat_keys)
        println(file, "  [$n 嵌段]")
        for (k, v) in final_stats[n]
            println(file, "    - $k: $v 样本")
        end
    end
end
println("\n统计报告已保存至: $txt_path")

println("\n开始将数据集按嵌段数量拆分并保存到子文件夹...")

# 为每个嵌段数量创建子文件夹并保存
dataset_keys = collect(keys(datasets))
Threads.@threads for n in sort(dataset_keys)
    output_dir = joinpath(output_base_dir, "$(n)blocks")
    mkpath(output_dir)
    
    data_list = datasets[n]
    total_to_save = length(data_list)
    
    for (i, data) in enumerate(data_list)
        F_pad, edge_index, edge_attr, χN_star, k_star, topo_type, chain = data

        sample_name = "sample_$(lpad(i, 5, "0"))"
        base_path = joinpath(output_dir, sample_name)

        # 1. 保存 MAT
        matopen("$(base_path).mat", "w") do file
            write(file, "F_pad", F_pad)
            write(file, "edge_index", edge_index)
            write(file, "edge_attr", edge_attr)
        end

        # 2. 提取 Blocks
        blocks_info = []
        for block in chain.blocks
            push!(blocks_info, Dict(
                "label" => string(block.label),
                "segment" => string(block.segment.label),
                "length" => block.f,
                "ends" => [string(block.E1.label), string(block.E2.label)]
            ))
        end

        # 3. 构建 YAML 字典
        config_dict = Dict(
            "system" => Dict(
                "χN_star" => χN_star,
                "k_star" => k_star,
                "topology" => topo_type,  
                "components" => [
                    Dict(
                        "type" => "OTD",
                        "blocks" => blocks_info
                    )
                ]
            )
        )

        # 4. 保存 YAML
        YAML.write_file("$(base_path).yml", config_dict)
    end
    
    println("  ✓ $n 嵌段: $total_to_save 个样本已保存至 $output_dir/")
end

println("\n✅ 所有样本已按嵌段数量分组保存!")
println("文件夹结构:")
for n in sort(dataset_keys)
    println("  - $(n)blocks/ → $(length(datasets[n])) 个样本")

end
