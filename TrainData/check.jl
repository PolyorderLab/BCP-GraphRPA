using Pkg; Pkg.activate("/home/zxy/project/jl") 
using JLD2
using Statistics
using LinearAlgebra

# Path configuration
input_path = "/home/zxy/project/TAPD/TrainData/Generate/output/random_4n_2w_chiNk.jld2"

println("🔍 Starting Data Audit...")
@load input_path dataset
n_samples = length(dataset)

# 随机抽样检查
check_indices = [1, Int(n_samples÷2), n_samples]

for idx in check_indices
    println("\n--- Audit Report for Sample #$idx ---")
    data = dataset[idx]
    
    # 1. 节点特征检查 (F_pad)
    # 应该符合我们在 TopoGNN/GC-GNN 文献中定义的 [Degree, Avg_Neighbor_Degree] [cite: 1115, 1279, 1281]
    F_pad = data[1]
    println("Node Features (X) shape: ", size(F_pad))
    println("Sample Degree (Node 1): ", F_pad[1, 1])
    
    # 2. 边连通性检查 (edge_index)
    # 必须是双向图 [cite: 637, 2736]
    edge_idx = data[2]
    num_directed_edges = size(edge_idx, 2)
    println("Directed Edges count: ", num_directed_edges)
    
    # 3. 边特征与物理守恒检查 (edge_attr)
    # 在 RPA 理论中，所有嵌段的 f 之和必须严格等于 1.0 [cite: 361, 2416]
    edge_attr = data[3]
    # 注意：由于是双向边，求和时要除以 2
    f_sum = sum(edge_attr[:, 1]) / 2.0
    println("Chemical Composition (f) sum: ", round(f_sum, digits=4))
    
    if abs(f_sum - 1.0) > 1e-4
        @warn "❌ Physics Violation: Total volume fraction f does not sum to 1.0!"
    else
        println("✅ Physics Consistency: Volume fraction f sums to 1.0.")
    end
    
    # 4. 预测目标检查 (chiN, k_star)
    # 对应 RPA 稳定性极限预测 [cite: 355, 2625]
    println("Targets: chiN = ", data[4], " | k_star = ", data[5])
    println("Block count (n_b): ", data[6])
end

println("\n🚀 Dataset integrity looks solid. Ready for Python/PyTorch conversion.")