using Pkg; Pkg.activate("/home/zxy/project/jl") 
using Roots
using Optim
using Polymer
using PolymerArchitecture
using Polyorder: RPA
using Random, LatinHypercubeSampling, JLD2, Statistics, Graphs
using LinearAlgebra

# 强制线性代数库单线程，防止与 Julia 线程冲突导致死锁
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

function generate_random_architecture(n_b::Int; f_min=0.05, rng=Random.GLOBAL_RNG)
    monomers = rand(rng, [:A, :B], n_b)
    if !(:A in monomers) monomers[rand(rng, 1:n_b)] = :A end
    if !(:B in monomers) monomers[rand(rng, 1:n_b)] = :B end
    shares = get_random_f(rng, n_b, f_min)
    node_connections = Dict{Symbol, Int}()
    touch!(n) = node_connections[n] = get(node_connections, n, 0) + 1
    configs = []
    push!(configs, (m=monomers[1], f=shares[1], n1=:n1, n2=:n2))
    touch!(:n1); touch!(:n2)
    available_nodes = [:n1, :n2]
    max_node_idx = 2
    for i in 2:n_b
        parent = rand(rng, available_nodes)
        max_node_idx += 1
        new_n = Symbol("n$max_node_idx")
        push!(configs, (m=monomers[i], f=shares[i], n1=parent, n2=new_n))
        touch!(parent); touch!(new_n)
        push!(available_nodes, new_n)
    end
    node_objs = Dict(n => (d > 1 ? BranchPoint(n) : FreeEnd(n)) for (n, d) in node_connections)
    blocks = [PolymerBlock(Symbol("b$i"), KuhnSegment(c.m), c.f, node_objs[c.n1], node_objs[c.n2]) for (i, c) in enumerate(configs)]
    return BlockCopolymer(:Random, blocks)
end

function polymer_to_gnn(chain; N_max::Int=20, D_feat::Int=3)
    blocks = chain.blocks
    N_blocks = length(blocks)
    graph = BlockCopolymerGraph(chain)
    Nv = Graphs.nv(graph)
    if Nv > N_max return nothing end
    node_types = zeros(Float32, Nv)
    for i in 1:Nv
        connected_edges = filter(e -> (Graphs.src(e) == i || Graphs.dst(e) == i), collect(Graphs.edges(graph)))
        a_count = 0
        for e in connected_edges
            edge_idx = findfirst(x -> x == e, collect(Graphs.edges(graph)))
            if !isnothing(edge_idx) && edge_idx <= N_blocks
                label_str = string(blocks[edge_idx].segment.label)
                if occursin("A", label_str) a_count += 1 end
            end
        end
        node_types[i] = a_count >= length(connected_edges) / 2 ? 1.0f0 : 0.0f0
    end
    F_pad = zeros(Float32, N_max, D_feat)
    degs = Graphs.degree(graph)
    for i in 1:Nv
        F_pad[i, 1] = Float32(degs[i])
        F_pad[i, 2] = degs[i] > 2 ? 1.0f0 : 0.0f0
        F_pad[i, 3] = node_types[i]
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

function run_generation_mixed_parallel(total_target=5005000)
    dataset = []
    data_lock = ReentrantLock()
    nb_stats = zeros(Int, 9) # 涵盖到下标 8
    
    num_threads = Threads.nthreads()
    println("🚀 多线程生产模式 | 范围: 2n-4n | 线程数: $num_threads | 目标: $total_target")
    start_time = time()

    @sync for t in 1:num_threads
        Threads.@spawn begin
            local_rng = MersenneTwister(42 + Threads.threadid())
            
            while true
                if lock(()->length(dataset) >= total_target, data_lock) break end

                n_b = rand(local_rng, 2:4) 
                
                if n_b <= 4
                    n_max_val = 24
                elseif n_b <= 6
                    n_max_val = 32
                else
                    n_max_val = 40 # 7n-8n
                end
                
                chain = generate_random_architecture(n_b; f_min=0.05, rng=local_rng)
                
                try
                    res = RPA.compute_stability_limit(chain; χN0=15.0, k2_max=100.0)
                    χN = res[1]
                    k_star = res[2]
                    
                    if 0.1 < χN < 5000.0
                        gnn_out = polymer_to_gnn(chain; N_max=n_max_val, D_feat=3)
                        
                        if !isnothing(gnn_out)
                            lock(data_lock) do
                                if length(dataset) < total_target
                                    push!(dataset, (gnn_out..., Float32(χN), Float32(k_star), Int32(n_b)))
                                    nb_stats[n_b] += 1
                                    
                                    current_count = length(dataset)
                                    if current_count % 100 == 0 # 打印频率适当调高
                                        elapsed = round(time() - start_time, digits=1)
                                        percent = round(current_count / total_target * 100, digits=2)
                                        sps = current_count / elapsed
                                        eta_h = floor(Int, (total_target - current_count) / sps / 3600)
                                        eta_m = floor(Int, ((total_target - current_count) / sps % 3600) / 60)
                                        
                                        stat_str = join(["$(n)n:$(nb_stats[n])" for n in 2:4], " | ")
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
    
    println("\n\n✅ 生成完成！最终分布：")
    for n in 2:4 println(" - $(n) 嵌段: $(nb_stats[n]) 样本") end
    return dataset
end

output_dir = "train_data_random_4n"
mkpath(output_dir)

total_samples = 4200
final_data = run_generation_mixed_parallel(total_samples) 

save_path = joinpath(output_dir, "random_4n_4k2_chiNk.jld2")
@save save_path dataset = final_data
println("\n💾 数据集已保存至: $save_path")