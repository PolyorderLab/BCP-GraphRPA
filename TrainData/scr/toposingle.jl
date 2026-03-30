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

function get_random_f(rng, num_segments, f_min)
    if f_min * num_segments >= 1.0
        return fill(1.0/num_segments, num_segments)
    end
    remaining = 1.0 - (num_segments * f_min)
    temp_pts = sort(rand(rng, num_segments - 1))
    gaps = [temp_pts[1]; diff(temp_pts); 1.0 - temp_pts[end]]
    return gaps .* remaining .+ f_min
end

# ================= 新增：自带"去冗余"的精确拓扑生成核心 =================
function generate_specific_architecture(rng=Random.GLOBAL_RNG)
    topo_types = [:Linear, :Comb, :Branch, :Star, :Dendrimer]
    selected_type = rand(rng, topo_types)
    
    configs = [] 
    
    if selected_type == :Linear
        n_b = rand(rng, 2:5) # 2到5嵌段线型
        for i in 1:n_b
            push!(configs, (n1=Symbol("n$i"), n2=Symbol("n$(i+1)")))
        end
        
    elseif selected_type == :Comb
        n_branches = rand(rng, 2:4) # 骨架上有2到4个接枝点
        # 构建主链(骨架)
        for i in 1:(n_branches + 1)
            push!(configs, (n1=Symbol("bb$i"), n2=Symbol("bb$(i+1)")))
        end
        # 在主链节点上附加侧链 
        for i in 1:n_branches
            push!(configs, (n1=Symbol("bb$(i+1)"), n2=Symbol("br$i")))
        end
        
    elseif selected_type == :Branch # 对应 H-shape 或 Pom-Pom
        arms_left = rand(rng, 2:3)
        arms_right = rand(rng, 2:3)
        push!(configs, (n1=:bb_start, n2=:bb_end))
        for i in 1:arms_left push!(configs, (n1=:bb_start, n2=Symbol("L$i"))) end
        for i in 1:arms_right push!(configs, (n1=:bb_end, n2=Symbol("R$i"))) end
        
    elseif selected_type == :Star
        n_arms = rand(rng, 3:6) 
        for i in 1:n_arms
            push!(configs, (n1=:center, n2=Symbol("arm$i")))
        end
        
    elseif selected_type == :Dendrimer
        functionality = 3 
        current_nodes = [:center]
        next_nodes = []
        node_id = 1
        for g in 1:2 
            for node in current_nodes
                branches = (g == 1) ? functionality : (functionality - 1)
                for b in 1:branches
                    new_n = Symbol("d_node$node_id")
                    node_id += 1
                    push!(configs, (n1=node, n2=new_n))
                    push!(next_nodes, new_n)
                end
            end
            current_nodes = next_nodes
            next_nodes = []
        end
    end
    
    node_connections = Dict{Symbol, Int}()
    for c in configs
        node_connections[c.n1] = get(node_connections, c.n1, 0) + 1
        node_connections[c.n2] = get(node_connections, c.n2, 0) + 1
    end

    deg2_nodes = [k for (k, v) in node_connections if v == 2]
    conflict_pairs = []
    for n in deg2_nodes
        connected_edges = Int[]
        for (i, c) in enumerate(configs)
            if c.n1 == n || c.n2 == n
                push!(connected_edges, i)
            end
        end
        if length(connected_edges) == 2
            push!(conflict_pairs, (connected_edges[1], connected_edges[2]))
        end
    end
    
    n_b = length(configs)
    monomers = Vector{Symbol}(undef, n_b)
    
    while true
        temp_monomers = rand(rng, [:A, :B], n_b)
        
        if !(:A in temp_monomers) temp_monomers[rand(rng, 1:n_b)] = :A end
        if !(:B in temp_monomers) temp_monomers[rand(rng, 1:n_b)] = :B end
        is_valid = true
        for (e1, e2) in conflict_pairs
            if temp_monomers[e1] == temp_monomers[e2]
                is_valid = false
                break
            end
        end
        
        if is_valid
            monomers = temp_monomers
            break
        end
    end
    
    shares = get_random_f(rng, n_b, 0.05)
    
    node_objs = Dict(n => (d > 1 ? BranchPoint(n) : FreeEnd(n)) for (n, d) in node_connections)
    blocks = [PolymerBlock(Symbol("b$i"), KuhnSegment(monomers[i]), shares[i], node_objs[c.n1], node_objs[c.n2]) for (i, c) in enumerate(configs)]
    
    return BlockCopolymer(selected_type, blocks; check=false), string(selected_type), n_b
end

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

function run_generation_mixed_parallel(total_target=20000)
    dataset = []
    data_lock = ReentrantLock()
    
    stats = Dict("Linear"=>0, "Comb"=>0, "Branch"=>0, "Star"=>0, "Dendrimer"=>0)
    
    num_threads = Threads.nthreads()
    println("多线程生产模式 |线程数: $num_threads | 目标: $total_target")
    start_time = time()

    @sync for t in 1:num_threads
        Threads.@spawn begin
            local_rng = MersenneTwister(42 + Threads.threadid())
            
            while true
                if lock(()->length(dataset) >= total_target, data_lock) break end
                
                chain, topo_type, n_b = generate_specific_architecture(local_rng)
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
                                    stats[topo_type] += 1
                                    
                                    current_count = length(dataset)
                                    if current_count % 100 == 0 
                                        elapsed = round(time() - start_time, digits=1)
                                        percent = round(current_count / total_target * 100, digits=2)
                                        sps = current_count / elapsed
                                        eta_h = floor(Int, (total_target - current_count) / sps / 3600)
                                        eta_m = floor(Int, ((total_target - current_count) / sps % 3600) / 60)
                                        
                                        stat_str = "L:$(stats["Linear"]) C:$(stats["Comb"]) B:$(stats["Branch"]) S:$(stats["Star"]) D:$(stats["Dendrimer"])"
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

output_dir = "/home/zxy/project/TAPD/TrainData/data/train_topo"
mkpath(output_dir)

total_samples = 100000
final_data, final_stats = run_generation_mixed_parallel(total_samples) 

txt_path = "/home/zxy/project/TAPD/TrainData/data/topology_summary.txt"
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