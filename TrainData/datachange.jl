using Pkg; Pkg.activate("/home/zxy/project/jl") 
using JLD2
using HDF5
using ProgressMeter

# Path configuration
input_path = "/home/zxy/project/TAPD/TrainData/Generate/output/random_4n_2w_chiNk.jld2"
bridge_path = "/home/zxy/project/TAPD-1/TrainData/Generate/output/data_bridge.h5"

println("📂 Loading JLD2 dataset...")
@load input_path dataset
n_samples = length(dataset)

println("🚀 Exporting $n_samples samples to HDF5 bridge...")
# Ensure output directory exists
mkpath(dirname(bridge_path))
h5open(bridge_path, "w") do file
    @showprogress for i in 1:n_samples
        # Unpack: (F_pad, edge_index, edge_attr, chiN, k_star, n_b)
        data = dataset[i]
        
        # Create a group per sample (0-indexed for Python convenience)
        g = create_group(file, "s$(i-1)")
        
        # Basic numerical arrays
        g["x"] = data[1]           # Node features
        g["edge_index"] = data[2]  # Graph connectivity
        g["edge_attr"] = data[3]   # Block features (f and type)
        
        # Targets and metadata
        g["chiN"] = Float32(data[4])
        g["k_star"] = Float32(data[5])
        g["n_b"] = Int32(data[6])
    end
end
println("✅ Export complete: $bridge_path")