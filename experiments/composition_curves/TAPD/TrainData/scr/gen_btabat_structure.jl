using Pkg
Pkg.activate("/home/zxy/project/jl")

using PolymerArchitectureMakie: visualize, BlockCopolymerGraph
using CairoMakie
using Graphs: nv, ne, degree, edges
using Polymer: block, PolymerBlock, BranchPoint, FreeEnd, KuhnSegment, BlockCopolymer

vis_dir = "/home/zxy/project/TAPD/Demo/outcome/phase_diagrams"
mkpath(vis_dir)

function build_btabat_vis(fA)
    # (BT)AB(AT) topology from Xie et al. Figure 4:
    # main path A1-A2-B1-B2, with B3 tethered at the A1/A2 junction
    # and A3 tethered at the B1/B2 junction.
    fa1 = 0.25 * fA
    fa2 = 0.25 * fA
    fa3 = 0.50 * fA
    fb1 = 0.15 * (1.0 - fA)
    fb2 = 0.15 * (1.0 - fA)
    fb3 = 0.70 * (1.0 - fA)

    sA = KuhnSegment(:A)
    sB = KuhnSegment(:B)

    tA = BranchPoint(:tA)
    AB = BranchPoint(:AB)
    tB = BranchPoint(:tB)

    A1 = PolymerBlock(:A1, sA, fa1 * 2.5, FreeEnd(:A1), tA)
    A2 = PolymerBlock(:A2, sA, fa2 * 2.5, tA,           AB)
    B1 = PolymerBlock(:B1, sB, fb1,       AB,           tB)
    B2 = PolymerBlock(:B2, sB, fb2,       tB,           FreeEnd(:B2))
    B3 = PolymerBlock(:B3, sB, fb3 * 0.55, tA,          FreeEnd(:B3))
    A3 = PolymerBlock(:A3, sA, fa3 * 2.5, tB,           FreeEnd(:A3))

    return BlockCopolymer(Symbol("BTABAT"), [A1, A2, B1, B2, B3, A3]; check=false)
end

function custom_edge_labels(bcg)
    labels = String[]
    for e in edges(bcg)
        push!(labels, string(block(e).label))
    end
    return labels
end

chain_vis = build_btabat_vis(0.15)
bcg = BlockCopolymerGraph(chain_vis)
elabels = custom_edge_labels(bcg)

fig = visualize(chain_vis;
    node_size=12.0,
    elabels=elabels,
    elabels_textsize=24.0,
    fig_width=700,
    fig_height=450
)

save(joinpath(vis_dir, "BTABAT_structure.png"), fig)
println("Saved BTABAT_structure.png")
println("  BTABAT: Nv=$(nv(bcg)), Ne=$(ne(bcg)), degrees=$(degree(bcg))")
