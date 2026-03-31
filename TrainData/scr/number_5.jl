using Pkg; Pkg.activate("/home/zxy/project/jl") 
using Roots
using Optim
using Polymer
using PolymerArchitecture
using Polyorder: RPA
using Random, LatinHypercubeSampling, Statistics, Graphs
using LinearAlgebra
using YAML
using MAT

BLAS.set_num_threads(1)

# ================= 新增：基础组件切分（引入线程专属rng防止冲突） =================
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

# ================= 新增：纯随机图论生长算法核心 =================
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
    fA = rand(rng, 10:90) / 100.0
    
    # [步骤二] 树状图(Tree)严格生长，数学上100%无环
    edges = Tuple{Int, Int}[]
    degrees = Dict{Int, Int}(1 => 0)
    active_nodes = [1]
    
    for new_node in 2:(total_blocks + 1)
        # 为了符合高分子物理，限制单个节点最多连 4 根线
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
    
    # [步骤三] 图着色：强制过滤无效的“同色线型相连”
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
    
    # 贪心着色：相邻线型边必须异色
    for i in 1:total_blocks
        forbidden = [edge_colors[j] for j in edge_adj[i] if isassigned(edge_colors, j)]
        choices = setdiff([:A, :B], forbidden)
        # 如果是复杂星型节点导致没得选，才随机分配；线型必定异色
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
        # 根据统计好的度数，精准判断是 FreeEnd 还是 BranchPoint
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

# ================= 原有：向GNN特征矩阵的提取映射 (保持不变) =================
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
        if idx <= N_blocks
            curr_block = blocks[idx]
            f_val = Float32(curr_block.f)
            label_str = string(curr_block.segment.label)
            type_val = occursin("A", label_str) ? 1.0f0 : 0.0f0
        else
            f_val, type_val = 0.0f0, 0.5f0
        end
        
        edge_index[1, 2idx-1], edge_index[2, 2idx-1] = u, v
        edge_attr[2idx-1, 1], edge_attr[2idx-1, 2] = f_val, type_val
        
        edge_index[1, 2idx], edge_index[2, 2idx] = v, u
        edge_attr[2idx, 1], edge_attr[2idx, 2] = f_val, type_val
    end
    
    return F_pad, edge_index, edge_attr
end

# ================= 多线程并行采样与 RPA 计算 (进度条逻辑稍作动态调整) =================
function run_generation_mixed_parallel(total_target=20000)
    dataset = []
    data_lock = ReentrantLock()
    
    stats = Dict{String, Int}() # 改为动态字典，以适应不确定名称的 Random 拓扑
    
    num_threads = Threads.nthreads()
    println("多线程生产模式 | 线程数: $num_threads | 目标: $total_target")
    start_time = time()

    @sync for t in 1:num_threads
        Threads.@spawn begin
            local_rng = MersenneTwister(42 + Threads.threadid()) # 绑定线程安全的随机种子
            
            while true
                if lock(()->length(dataset) >= total_target, data_lock) break end
                
                target_blocks = rand(local_rng, 2:5) 
                chain, topo_type, n_b = generate_random_architecture(local_rng, target_blocks; minimum_f=0.03)
                n_max_val = 32 
                
                try
                    res = RPA.compute_stability_limit(chain; χN0=15.0, k2_max=100.0)
                    χN_star = res[1]
                    k_star = res[2]
                    
                    if 0.1 < χN_star < 5000.0
                        gnn_out = polymer_to_gnn(chain; N_max=n_max_val, D_feat=3)
                        
                        if !isnothing(gnn_out)
                            lock(data_lock) do
                                if length(dataset) < total_target
                                    push!(dataset, (gnn_out..., Float32(χN_star), Float32(k_star), topo_type, chain))
                                    stats[topo_type] = get(stats, topo_type, 0) + 1
                                    
                                    current_count = length(dataset)
                                    if current_count % 100 == 0 
                                        elapsed = round(time() - start_time, digits=1)
                                        percent = round(current_count / total_target * 100, digits=2)
                                        sps = current_count / elapsed
                                        eta_h = floor(Int, (total_target - current_count) / sps / 3600)
                                        eta_m = floor(Int, ((total_target - current_count) / sps % 3600) / 60)
                                        
                                        stat_str = join(["$k:$(v)" for (k,v) in stats], " ")
                                        print("\r\x1b[K>> $percent% ($current_count/$total_target) | ETA: $(eta_h)h$(eta_m)m | $stat_str")
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
    
    println("\n\n 生成完成！拓扑分布：")
    for (k, v) in stats
        println(" - $k: $v 样本")
    end
    return dataset, stats
end

output_dir = "/home/zxy/project/TAPD/TrainData/data/train_nnumber/"
mkpath(output_dir)

total_samples = 100000
final_data, final_stats = run_generation_mixed_parallel(total_samples) 

txt_path = "/home/zxy/project/TAPD/TrainData/data/nnumber_summary.txt"
open(txt_path, "w") do file
    println(file, "生成完成！拓扑分布：")
    for (k, v) in final_stats
        println(file, " - $k: $v 样本")
    end
end
println("\n统计报告已保存至: $txt_path") 

println("\n💾 开始将数据集拆分为 YML 配置 和 MAT 数据张量 (⚡ 多线程加速保存中)...")

# 创建一个线程安全的原子计数器，用于记录保存进度
save_counter = Threads.Atomic{Int}(0)
total_to_save = length(final_data)

# 使用 Threads.@threads 开启多线程并行 for 循环
Threads.@threads for i in 1:total_to_save
    data = final_data[i]
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

    # --- 打印进度条逻辑 ---
    Threads.atomic_add!(save_counter, 1)
    curr = save_counter[]
    # 每保存完 500 个样本打印一次，避免多线程频繁抢占屏幕输出导致日志混乱
    if curr % 500 == 0 || curr == total_to_save
        percent = round(curr / total_to_save * 100, digits=2)
        print("\r\x1b[K>> 正在写入硬盘: $percent% ($curr/$total_to_save)")
        flush(stdout)
    end
end

println("\n\n✅ 速度起飞！所有样本的 .yml 与 .mat 文件已成功拆分并多线程保存至: $output_dir")