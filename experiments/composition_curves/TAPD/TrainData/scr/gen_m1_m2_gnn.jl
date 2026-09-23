using Pkg; Pkg.activate("/home/zxy/project/jl")
using Random
using Polymer
using PolymerArchitecture
using LinearAlgebra
using Graphs

# =================================================================
# 手动计算 GNN 特征（绕过 BlockCopolymerGraph，避免 SimpleGraph 合并边）
# =================================================================
function polymer_to_gnn_manual(blocks; N_max::Int=32, D_feat::Int=3)
    # Step 1: build node list from blocks
    # each block has two ends (FreeEnd or BranchPoint), each gets a unique node index
    # Same node object = same physical node
    node_list = []
    node_idx = Dict{Any, Int}()
    function get_node_idx(node)
        if !haskey(node_idx, node)
            push!(node_list, node)
            node_idx[node] = length(node_list)
        end
        return node_idx[node]
    end

    # Collect all nodes (deduplicated by object identity)
    for b in blocks
        get_node_idx(b.E1)
        get_node_idx(b.E2)
    end

    Nv = length(node_list)
    if Nv > N_max
        return nothing
    end

    # Step 2: build directed edge list from blocks
    # Each block creates one directed edge in each direction
    directed_edges = Tuple{Int,Int}[]  # (src, dst), 1-indexed
    block_to_edge = Tuple{Int,Int}[]  # undirected key (smaller, larger)
    for b in blocks
        u = node_idx[b.E1]
        v = node_idx[b.E2]
        push!(directed_edges, (u, v))
        push!(directed_edges, (v, u))
        key = u < v ? (u, v) : (v, u)
        push!(block_to_edge, key)
    end

    # Step 3: compute node features
    deg = zeros(Int, Nv)
    for (u, v) in directed_edges
        deg[u] += 1
    end

    # build adjacency list
    adj = [Int[] for _ in 1:Nv]
    for (u, v) in directed_edges
        push!(adj[u], v)
    end

    F_pad = zeros(Float32, N_max, D_feat)
    for i in 1:Nv
        F_pad[i, 1] = Float32(deg[i])
        nbrs = adj[i]
        if !isempty(nbrs)
            avg_deg = sum(deg[n] for n in nbrs) / length(nbrs)
            F_pad[i, 2] = Float32(avg_deg)
        else
            F_pad[i, 2] = 0.0f0
        end
    end

    # Step 4: edge features
    num_directed = length(directed_edges)
    edge_index = zeros(Int64, num_directed, 2)
    edge_attr = zeros(Float32, num_directed, 2)

    for (i, (u, v)) in enumerate(directed_edges)
        edge_index[i, 1] = u
        edge_index[i, 2] = v
        # find which block this corresponds to (by matching the undirected key)
        key = u < v ? (u, v) : (v, u)
        bidx = findfirst(==(key), block_to_edge)
        b = blocks[bidx]
        f_val = Float32(b.f)
        label_str = string(b.segment.label)
        type_val = occursin("A", label_str) ? 1.0f0 : 0.0f0
        edge_attr[i, 1] = f_val
        edge_attr[i, 2] = type_val
    end

    return F_pad, edge_index, edge_attr
end

# Legacy wrapper for compatibility with other topologies
function polymer_to_gnn(chain; N_max::Int=32, D_feat::Int=3)
    return polymer_to_gnn_manual(chain.blocks; N_max=N_max, D_feat=D_feat)
end

# =================================================================
# 手动用 Polymer.jl API 构建 M1 和 M2
# =================================================================
function build_m1(fA)
    # M1: 中心点eb0连接所有臂，eb2是分叉点
    #   A1 = (FreeEnd, eb0)
    #   B1 = (eb0, eb1)
    #   B2 = (eb0, eb2)
    #   B3 = (eb0, eb3)
    #   A2 = (eb2, FreeEnd)
    #
    # tauB1=0.05, tauB2=0.25, tauB3=0.70

    fB1 = 0.05 * (1.0 - fA)
    fB2 = 0.25 * (1.0 - fA)
    fB3 = 0.70 * (1.0 - fA)
    fa  = fA / 2.0

    sA = KuhnSegment(:A)
    sB = KuhnSegment(:B)

    eb0 = BranchPoint(:eb0)  # 中心点
    eb1 = BranchPoint(:eb1)  # B1终点
    eb2 = BranchPoint(:eb2)  # B2终点 / A2起点
    eb3 = BranchPoint(:eb3)  # B3终点

    # A1: FreeEnd <-> eb0
    A1 = PolymerBlock(:A1, sA, fa,  FreeEnd(:A1), eb0)
    # B1: eb0 <-> eb1
    B1 = PolymerBlock(:B1, sB, fB1, eb0,           eb1)
    # B2: eb0 <-> eb2
    B2 = PolymerBlock(:B2, sB, fB2, eb0,           eb2)
    # B3: eb0 <-> eb3
    B3 = PolymerBlock(:B3, sB, fB3, eb0,           eb3)
    # A2: eb2 <-> FreeEnd
    A2 = PolymerBlock(:A2, sA, fa,  eb2,           FreeEnd(:A2))

    return BlockCopolymer(Symbol("M1"), [A1, B1, B2, B3, A2]; check=false)
end

function build_m2(fA)
    # M2: 中心点eb0连接所有臂，eb2是分叉点
    #   A1 = (FreeEnd, eb0)
    #   B1 = (eb0, eb1)
    #   B2 = (eb0, eb2)
    #   A2 = (eb2, FreeEnd)
    #   (没有B3)
    #
    # τB2=0.26, τB1=1-τB2=0.74

    fB1 = 0.74 * (1.0 - fA)  # τB1 = 0.74
    fB2 = 0.26 * (1.0 - fA)
    fa  = fA / 2.0

    sA = KuhnSegment(:A)
    sB = KuhnSegment(:B)

    eb0 = BranchPoint(:eb0)  # 中心点
    eb1 = BranchPoint(:eb1)  # B1终点
    eb2 = BranchPoint(:eb2)  # B2终点 / A2起点

    # A1: FreeEnd <-> eb0
    A1 = PolymerBlock(:A1, sA, fa,  FreeEnd(:A1), eb0)
    # B1: eb0 <-> eb1
    B1 = PolymerBlock(:B1, sB, fB1, eb0,           eb1)
    # B2: eb0 <-> eb2
    B2 = PolymerBlock(:B2, sB, fB2, eb0,           eb2)
    # A2: eb2 <-> FreeEnd
    A2 = PolymerBlock(:A2, sA, fa,  eb2,           FreeEnd(:A2))

    return BlockCopolymer(Symbol("M2"), [A1, B1, B2, A2]; check=false)
end

# =================================================================
# 输出 GNN 特征到 JSON（供 Python 读取）
# =================================================================
using JSON

output = Dict()

for topo in ["M1", "M2"]
    chains = topo == "M1" ? [build_m1(fA) for fA in LinRange(0.11, 0.19, 100)] :
                            [build_m2(fA) for fA in LinRange(0.11, 0.19, 100)]

    fA_list = [fA for fA in LinRange(0.11, 0.19, 100)]
    F_list = []
    EI_list = []
    EA_list = []

    for (i, chain) in enumerate(chains)
        fA_val = fA_list[i]
            gnn = polymer_to_gnn(chain; N_max=32, D_feat=3)

        if !isnothing(gnn)
            F_pad, edge_index, edge_attr = gnn
            push!(F_list, map(Vector{Float32}, eachrow(F_pad)))
            push!(EI_list, map(Vector{Int64}, eachcol(edge_index)))
            push!(EA_list, map(Vector{Float32}, eachrow(edge_attr)))
        else
            println("WARNING: $topo fA=$fA_val has Nv > 32, skipping")
        end
    end

    output[string(topo)] = Dict(
        "fA" => fA_list,
        "F" => F_list,
        "edge_index" => EI_list,
        "edge_attr" => EA_list
    )
    println("Processed $(length(F_list)) samples for $topo")
end

json_path = "/home/zxy/project/TAPD/Demo/outcome/phase_diagrams/m1_m2_gnn_features.json"
json_str = JSON.json(output)
open(json_path, "w") do f
    write(f, json_str)
end
println("\nSaved GNN features to: $json_path")

# =================================================================
# 可视化验证（和 visua.jl 完全一样的流程）
# =================================================================
using PolymerArchitectureMakie: visualize, BlockCopolymerGraph
using CairoMakie
using Graphs: nv, ne, degree, edges, src, dst
using Polymer: block, block_length, specie, PolymerBlock, BranchPoint, FreeEnd, KuhnSegment, BlockCopolymer

vis_dir = "/home/zxy/project/TAPD/Demo/outcome/phase_diagrams"
mkpath(vis_dir)

function build_m1_vis(fA)
    # 可视化用的M1，A1和A2更长
    fB1 = 0.05 * (1.0 - fA)
    fB2 = 0.25 * (1.0 - fA)
    fB3 = 0.70 * (1.0 - fA)
    fa  = fA / 2.0

    sA = KuhnSegment(:A)
    sB = KuhnSegment(:B)

    eb0 = BranchPoint(:eb0)
    eb1 = BranchPoint(:eb1)
    eb2 = BranchPoint(:eb2)
    eb3 = BranchPoint(:eb3)

    # A1和A2更长（用于可视化）
    A1 = PolymerBlock(:A1, sA, fa * 2.5, FreeEnd(:A1), eb0)
    B1 = PolymerBlock(:B1, sB, fB1,      eb0,           eb1)
    B2 = PolymerBlock(:B2, sB, fB2,      eb0,           eb2)
    B3 = PolymerBlock(:B3, sB, fB3,      eb0,           eb3)
    A2 = PolymerBlock(:A2, sA, fa * 2.5, eb2,           FreeEnd(:A2))

    return BlockCopolymer(Symbol("M1"), [A1, B1, B2, B3, A2]; check=false)
end

function build_m2_vis(fA)
    # 可视化用的M2，A1和A2更长
    fB1 = 0.74 * (1.0 - fA)
    fB2 = 0.26 * (1.0 - fA)
    fa  = fA / 2.0

    sA = KuhnSegment(:A)
    sB = KuhnSegment(:B)

    eb0 = BranchPoint(:eb0)
    eb1 = BranchPoint(:eb1)
    eb2 = BranchPoint(:eb2)

    # A1和A2更长（用于可视化）
    A1 = PolymerBlock(:A1, sA, fa * 2.5, FreeEnd(:A1), eb0)
    B1 = PolymerBlock(:B1, sB, fB1,      eb0,           eb1)
    B2 = PolymerBlock(:B2, sB, fB2,      eb0,           eb2)
    A2 = PolymerBlock(:A2, sA, fa * 2.5, eb2,           FreeEnd(:A2))

    return BlockCopolymer(Symbol("M2"), [A1, B1, B2, A2]; check=false)
end

for topo in ["M1", "M2"]
    chain_vis = topo == "M1" ? build_m1_vis(0.15) : build_m2_vis(0.15)
    bcg = BlockCopolymerGraph(chain_vis)

    # 创建自定义的edge_labels函数，只显示标签名不显示分数
    function custom_edge_labels(bcg)
        labels = String[]
        for e in edges(bcg)
            label = string(block(e).label)
            push!(labels, label)
        end
        return labels
    end

    elabels = custom_edge_labels(bcg)

    # 使用统一的尺寸画图
    fig = visualize(chain_vis;
        node_size=12.0,
        elabels=elabels,
        elabels_textsize=24.0,
        fig_width=700,
        fig_height=450
    )
    save(joinpath(vis_dir, "$(topo)_structure.png"), fig)
    println("Saved $(topo)_structure.png")

    # 打印图的基本信息
    Nv = nv(bcg)
    Ne = ne(bcg)
    degs = degree(bcg)
    println("  $topo: Nv=$Nv, Ne=$Ne, degrees=$degs")
end

println("\nDone!")
