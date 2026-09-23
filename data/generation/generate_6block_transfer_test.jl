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
using JLD2

BLAS.set_num_threads(1)

function parse_args()
    opts = Dict{String, String}()
    i = 1
    while i <= length(ARGS)
        key = ARGS[i]
        if startswith(key, "--")
            if i == length(ARGS) || startswith(ARGS[i + 1], "--")
                opts[key] = "true"
                i += 1
            else
                opts[key] = ARGS[i + 1]
                i += 2
            end
        else
            i += 1
        end
    end
    return opts
end

function get_chain_signature(chain::BlockCopolymer, topo_type::String)
    seg_types = Int[]
    for b in chain.blocks
        label = string(b.segment.label)
        push!(seg_types, occursin("A", label) ? 1 : 0)
    end
    f_vals = round.([b.f for b in chain.blocks], digits=4)
    return "$(topo_type)_$(length(chain.blocks))_$(join(seg_types, ","))_$(join(f_vals, ","))"
end

function get_random_f(rng::AbstractRNG, num_segments::Int, total_f::Float64, f_min::Float64)
    if num_segments == 0
        return Float64[]
    end
    if num_segments == 1
        return [total_f]
    end
    if f_min * num_segments >= total_f
        return fill(total_f / num_segments, num_segments)
    end
    remaining = total_f - num_segments * f_min
    pts = sort(rand(rng, num_segments - 1))
    gaps = [pts[1]; diff(pts); 1.0 - pts[end]]
    return gaps .* remaining .+ f_min
end

function generate_random_architecture(rng::AbstractRNG, total_blocks::Int; minimum_f=0.03)
    fA = rand(rng) * 0.90 + 0.05
    edges = Tuple{Int, Int}[]
    degrees = Dict{Int, Int}(1 => 0)
    active_nodes = [1]

    for new_node in 2:(total_blocks + 1)
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

    edge_colors = Vector{Symbol}(undef, total_blocks)
    edge_adj = [Int[] for _ in 1:total_blocks]
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

    for i in 1:total_blocks
        forbidden = [edge_colors[j] for j in edge_adj[i] if isassigned(edge_colors, j)]
        choices = setdiff([:A, :B], forbidden)
        edge_colors[i] = isempty(choices) ? rand(rng, [:A, :B]) : rand(rng, choices)
    end
    if !(:A in edge_colors)
        edge_colors[rand(rng, 1:total_blocks)] = :A
    end
    if !(:B in edge_colors)
        edge_colors[rand(rng, 1:total_blocks)] = :B
    end

    num_A = count(==(:A), edge_colors)
    num_B = total_blocks - num_A
    f_A_list = get_random_f(rng, num_A, fA, minimum_f)
    f_B_list = get_random_f(rng, num_B, 1.0 - fA, minimum_f)

    blocks = PolymerBlock[]
    idx_A, idx_B = 1, 1
    for i in 1:total_blocks
        u, v = edges[i]
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
        push!(blocks, PolymerBlock(Symbol("b$i"), KuhnSegment(color), f_val, node_u, node_v))
    end

    chain = BlockCopolymer(Symbol("AB"), blocks; check=false)
    return chain, "Random_$(total_blocks)blocks"
end

function polymer_to_gnn(chain; N_max::Int=32, D_feat::Int=3)
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
            F_pad[i, 2] = Float32(sum(degs[n] for n in neighbor_nodes) / length(neighbor_nodes))
        end
    end

    edges_obj = collect(Graphs.edges(graph))
    edge_index = zeros(Int64, 2, length(edges_obj) * 2)
    edge_attr = zeros(Float32, length(edges_obj) * 2, 2)

    for (idx, e) in enumerate(edges_obj)
        u, v = Graphs.src(e), Graphs.dst(e)
        edge_key = u < v ? (u, v) : (v, u)
        if haskey(graph.edge2block, edge_key)
            block = graph.edge2block[edge_key]
            f_val = Float32(block.f)
            type_val = occursin("A", string(block.segment.label)) ? 1.0f0 : 0.0f0
        else
            f_val, type_val = 0.0f0, 0.5f0
        end
        edge_index[1, 2idx - 1], edge_index[2, 2idx - 1] = u, v
        edge_attr[2idx - 1, 1], edge_attr[2idx - 1, 2] = f_val, type_val
        edge_index[1, 2idx], edge_index[2, 2idx] = v, u
        edge_attr[2idx, 1], edge_attr[2idx, 2] = f_val, type_val
    end
    return F_pad, edge_index, edge_attr
end

function parse_block_counts(value::String)
    return [parse(Int, strip(x)) for x in split(value, ",") if !isempty(strip(x))]
end

function save_sample(output_dir::String, block_count::Int, sample_idx::Int, data)
    F_pad, edge_index, edge_attr, chiN_star, k_star, topo_type, chain = data
    sample_name = "$(block_count)blocks_sample_$(lpad(sample_idx, 5, "0"))"
    base_path = joinpath(output_dir, sample_name)

    jldsave("$(base_path).mat"; F_pad, edge_index, edge_attr)

    blocks_info = []
    for block in chain.blocks
        push!(blocks_info, Dict(
            "label" => string(block.label),
            "segment" => string(block.segment.label),
            "length" => block.f,
            "ends" => [string(block.E1.label), string(block.E2.label)]
        ))
    end

    YAML.write_file("$(base_path).yml", Dict(
        "system" => Dict(
            "χN_star" => chiN_star,
            "k_star" => k_star,
            "topology" => topo_type,
            "components" => [Dict("type" => "OTD", "blocks" => blocks_info)]
        )
    ))
end

function generate_block_transfer_set(; block_counts::Vector{Int}=[6],
    count_per_block::Int=500,
    output_dir::String=joinpath(@__DIR__, "..", "datasets", "6to10blocks_500_each"),
    seed::Int=20260707,
    minimum_f::Float64=0.03,
    chiN_min::Float64=0.1,
    chiN_max::Float64=1000.0)

    count_per_block > 0 || error("Count must be positive")
    ispath(output_dir) && error("Output already exists: $output_dir")
    mkpath(output_dir)
    rng = MersenneTwister(seed)
    seen = Dict(n => Set{String}() for n in block_counts)
    accepted = Dict(n => 0 for n in block_counts)
    attempted = Dict(n => 0 for n in block_counts)
    skipped_duplicate = Dict(n => 0 for n in block_counts)
    skipped_range = Dict(n => 0 for n in block_counts)
    skipped_error = Dict(n => 0 for n in block_counts)
    start_time = time()

    total_target = count_per_block * length(block_counts)
    println("Generating block-count transfer test set")
    println("  block_counts    = $(join(block_counts, ","))")
    println("  count_per_block = $count_per_block")
    println("  total_target    = $total_target")
    println("  output_dir      = $output_dir")
    println("  chiN range      = ($chiN_min, $chiN_max)")

    for block_count in block_counts
        println("\n[$(block_count) blocks]")
        while accepted[block_count] < count_per_block
            attempted[block_count] += 1
            chain, topo_type = generate_random_architecture(rng, block_count; minimum_f=minimum_f)
            sig = get_chain_signature(chain, topo_type)
            if sig in seen[block_count]
                skipped_duplicate[block_count] += 1
                continue
            end
            push!(seen[block_count], sig)

            try
                res = RPA.compute_stability_limit(chain; χN0=15.0, k2_max=100.0)
                chiN_star = Float32(res[1])
                k_star = Float32(res[2])
                if !(chiN_min < chiN_star < chiN_max)
                    skipped_range[block_count] += 1
                    continue
                end
                gnn_out = polymer_to_gnn(chain; N_max=32, D_feat=3)
                if isnothing(gnn_out)
                    skipped_error[block_count] += 1
                    continue
                end
                accepted[block_count] += 1
                save_sample(
                    output_dir,
                    block_count,
                    accepted[block_count],
                    (gnn_out..., chiN_star, k_star, topo_type, chain),
                )

                if accepted[block_count] % 25 == 0 || accepted[block_count] == count_per_block
                    elapsed = round(time() - start_time, digits=1)
                    total_accepted = sum(values(accepted))
                    println(
                        "  accepted $(accepted[block_count])/$count_per_block for $(block_count)blocks | " *
                        "total=$total_accepted/$total_target | attempted=$(attempted[block_count]) | " *
                        "range_skip=$(skipped_range[block_count]) | elapsed=$(elapsed)s"
                    )
                end
            catch err
                skipped_error[block_count] += 1
                if skipped_error[block_count] <= 5
                    @warn "RPA/sample generation failed" block_count=block_count exception=(err, catch_backtrace())
                end
                continue
            end
        end
    end

    summary_path = joinpath(output_dir, "generation_summary.yml")
    per_block = Dict(
        string(n) => Dict(
            "accepted" => accepted[n],
            "attempted" => attempted[n],
            "skipped_duplicate" => skipped_duplicate[n],
            "skipped_range" => skipped_range[n],
            "skipped_error" => skipped_error[n],
        ) for n in block_counts
    )
    YAML.write_file(summary_path, Dict(
        "block_counts" => block_counts,
        "count_per_block" => count_per_block,
        "total_target" => total_target,
        "accepted_total" => sum(values(accepted)),
        "seed" => seed,
        "minimum_f" => minimum_f,
        "chiN_min" => chiN_min,
        "chiN_max" => chiN_max,
        "per_block" => per_block,
        "output_dir" => output_dir
    ))
    println("Done. Summary: $summary_path")
end

opts = parse_args()
block_counts = parse_block_counts(get(opts, "--block-counts", "6"))
count_per_block = parse(Int, get(opts, "--count-per-block", get(opts, "--count", "500")))
default_output = length(block_counts) == 1 && block_counts[1] == 6 ?
    joinpath(@__DIR__, "..", "datasets", "6blocks_500") :
    joinpath(@__DIR__, "..", "datasets", "$(minimum(block_counts))to$(maximum(block_counts))blocks_$(count_per_block)_each")
generate_block_transfer_set(
    block_counts=block_counts,
    count_per_block=count_per_block,
    output_dir=get(opts, "--output-dir", default_output),
    seed=parse(Int, get(opts, "--seed", "20260707")),
    minimum_f=parse(Float64, get(opts, "--minimum-f", "0.03")),
    chiN_min=parse(Float64, get(opts, "--chiN-min", "0.1")),
    chiN_max=parse(Float64, get(opts, "--chiN-max", "1000.0")),
)
