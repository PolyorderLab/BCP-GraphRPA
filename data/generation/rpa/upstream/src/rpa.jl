"""
    RPA

Random Phase Approximation (RPA) module for analyzing polymer phase behavior and computing 
structure factors, form factors, and stability limits for polymer systems.

This module provides functions for:
- Computing single chain structure factors and form factors
- Analyzing spinodal stability limits
- Calculating RPA structure factors for multi-component polymer systems
- Computing scattering structure factor matrices
"""
module RPA

using Graphs: Graphs, a_star, dst, src
using LinearAlgebra: LinearAlgebra, /, I
using Polymer: Polymer, BlockCopolymer, PolymerBlock, PolymerSystem,
                SpecieNumberType, TwoSpeciesSystem, nspecies, specie,
                specie_number_type, species
using PolymerArchitecture: PolymerArchitecture, BlockCopolymerGraph

@doc raw"""
    _h(x)

This function evaluate $\frac{1-e^{-x}}{x}$ for all $x$ to an error down to 1.02 ULP.

See https://discourse.julialang.org/t/is-there-an-implementation-of-trefethens-contour-integrals-for-evaluating-1-exp-x-x-as-x-0/57680
"""
_h(x) = iszero(x) ? float(one(x)) : -expm1(-x)/x

function _g(x)
    return evalpoly(x, (1, -1/3, 1/12, -1/60, 1/360, -1/2520, 1/20160, -1/181440, 1/1814400))
end

# const t = Taylor1(6)
# const taylorh = (1 - exp(-t))/t

@doc raw"""
    g(x, f=1.0)

Evaluate the Debye function
    $g(x, f) = \frac{2}/{x^2} \left(e^{-xf} + xf - 1 \right)$
where $x = k^2R_g^2$.
"""
function g(x, f=1.0)
    # The limiting value when x = 0.
    iszero(x) && return f^2

    # For large xf, there is no cancellation error.
    xf = x * f
    abs(xf) > 0.1 && return 2(exp(-xf) + xf - 1) / x^2

    # Use hard coded taylor expansion for small xf to eliminate the cancellation error.
    return f^2 * _g(xf)
end

@doc raw"""
    h(x, f=1.0)

Evaluate the $h$ function for RPA approximation
    $h(x, f) = \frac{1 - e^{-xf}}{x}$
where $x = k^2R_g^2$.
"""
function h(x, f=1.0)
    xf = x * f
    return f * _h(xf)
end

@doc raw"""
    l(x, f=1.0)

Evaluate the $l$ function for RPA approximation
    $l(x, f) = e^{-xf}$
where $x = k^2R_g^2$.
"""
l(x, f=1.0) = exp(-x * f)

function form_factor(bcg::BlockCopolymerGraph, block1::PolymerBlock, block2::PolymerBlock, k2::AbstractArray)
    e1 = bcg.block2edge[block1]
    e2 = bcg.block2edge[block2]

    # for same blocks
    (e1 == e2) && return g.(k2, block1.f)

    # For blocks have common block ends
    if length(Set([e1..., e2...])) == 3
        return h.(k2, block1.f) .* h.(k2, block2.f)
    end

    # For blocks are at least one block apart
    d = zero(block1.f)
    for e in a_star(bcg.graph, e1[1], e2[1])
        # Do not count block1 and block2
        et = Polymer._sort_tuple2( (src(e), dst(e)) )
        (et == e1) && continue
        (et == e2) && continue
        block = bcg.edge2block[et]
        d += block.f
    end
    return h.(k2, block1.f) .* l.(k2, d) .* h.(k2, block2.f)
end

@doc raw"""
    form_factor(chain::BlockCopolymer, k2::Real)
    form_factor(chain::BlockCopolymer, k2::AbstractArray)

Compute the form factor of a block copolymer chain for scattering vector(s) using the RPA approximation.

The form factor is
    $P(k) = \sum_i \sum_j g_{ij}(k)$
where $i$ and $j$ iterates all blocks of the block copolymer.
    $g_{ij}(k) = g_D(k, f_i)$
when $i = j$,
    $g_{ij}(k) = h(k, f_i) h(k, f_j)$
when $i$ and $j$ have common block ends, and
    $g_{ij}(k) = h(k, f_i) l(k,d_{ij}) h(k, f_j)$
when $i$ and $j$ are at least one block apart.
Here $d_{ij}$ is the sum of the lengths of blocks that connect block $i$ and $j$ with shortest path.
"""
function form_factor(chain::BlockCopolymer, k2::AbstractArray)
    bcg = BlockCopolymerGraph(chain)
    P = zero(k2)
    for block1 in chain.blocks
        for block2 in chain.blocks
            P .+= form_factor(bcg, block1, block2, k2)
        end
    end
    return P
end

function form_factor(chain::BlockCopolymer, k2::Real)
    return form_factor(chain, [k2])[1]
end

@doc raw"""
    form_factor(chain::BlockCopolymer, specie1::Symbol, specie2::Symbol k2::Real)
    form_factor(chain::BlockCopolymer, specie1::Symbol, specie2::Symbol, k2::AbstractArray)

Return a partial form factor of a block copolymer chain for two specified species. Suppose the specie1 is A, adn specie2 is B, then $P_{AB}(k^2)$ is computed.
"""
function form_factor(chain::BlockCopolymer, sp1::Symbol, sp2::Symbol, k2::AbstractArray)
    bcg = BlockCopolymerGraph(chain)
    P = zero(k2)
    for block1 in chain.blocks
        for block2 in chain.blocks
            if specie(block1) == sp1 && specie(block2) == sp2
                P .+= form_factor(bcg, block1, block2, k2)
            end
        end
    end
    return P
end

function form_factor(chain::BlockCopolymer, sp1::Symbol, sp2::Symbol, k2::Real)
    return form_factor(chain, sp1, sp2, [k2])[1]
end

@doc raw"""
    form_factor(chain::BlockCopolymer, specie::Symbol, k2::Real)
    form_factor(chain::BlockCopolymer, specie::Symbol, k2::AbstractArray)

Return a partial form factor of a block copolymer chain for a specified specie. Suppose the specie is A, then $P_{AA}(k^2)$ is computed.
"""
function form_factor(chain::BlockCopolymer, sp::Symbol, k2::AbstractArray)
    return form_factor(chain, sp, sp, k2)
end

function form_factor(chain::BlockCopolymer, sp::Symbol, k2::Real)
    return form_factor(chain, sp, [k2])[1]
end

@doc raw"""
    reciprocal_structure_factor(chain::BlockCopolymer, χN, k2::Real)
    reciprocal_structure_factor(chain::BlockCopolymer, χN, k2::AbstractVector)

Return the reciprocal structure factor: $N/S$.

See also [`RPA.structure_factor`](@ref).
"""
function reciprocal_structure_factor(chain::BlockCopolymer, χN, k2::AbstractVector{<:Real}=0:0.01:5.0)
    A, B = species(chain)
    S_AA = form_factor(chain, A, k2)
    S_BB = form_factor(chain, B, k2)
    S_AB = form_factor(chain, A, B, k2)
    S = S_AA + S_BB + 2S_AB
    W = S_AA .* S_BB - S_AB .* S_AB

    return @. S/W - 2χN
end

function reciprocal_structure_factor(chain::BlockCopolymer, χN, k2::Real)
    return reciprocal_structure_factor(chain, χN, [k2])[1]
end

@doc raw"""
    structure_factor(chain::BlockCopolymer, χN, k2::Real)
    structure_factor(chain::BlockCopolymer, χN, k2::AbstractVector)

Compute the structure factor (scattering factor) of a *two-specie* block copolymer melt:
    $S = W / (S^t - 2χW)$
where $S^t$ is the sum of all elemments of matrix $S_{ij}$ where $i,j=A,B$, and $W$ is the determinant of the matrix. Note that here $S/N$ is actually returned.

# References

* Leibler, L. Theory of Microphase Separation in Block Copolymers. Macromolecules 1980, 13 (6), 1602–1617.
* Ranjan, A. Theoretical Studies of Equilibrium Structure and Linear Response in Diblock Copolymer Melts, University of Minnesota, 2010. https://doi.org/10.1002/9781119990413.ch1.
* Ranjan, A.; Qin, J.; Morse, D. C. Linear Response and Stability of Ordered Phases of Block Copolymer Melts. Macromolecules 2008, 41 (3), 942–954.
"""
function structure_factor(chain::BlockCopolymer, χN, k2::AbstractVector{<:Real}=0:0.01:5.0)
    return 1 ./ reciprocal_structure_factor(chain, χN, k2)
end

function structure_factor(chain::BlockCopolymer, χN, k2::Real)
    return structure_factor(chain, χN, [k2])[1]
end

@doc raw"""
    compute_Smat(bcp::BlockCopolymer, k2::Real)
    compute_Smat(bcp::BlockCopolymer, k2::AbstractArray)
    compute_Smat!(Smat::AbstractMatrix, bcp::BlockCopolymer, k2::AbstractArray)

Compute the scattering matrix (linear response of an ideal gas of a block copolymer chain) $S$ of a BlockCopolymer instance. $S_{ij}$ is the element of matrix $S$, where $i, j$ are the index of specie type. Take AB diblock copolymer for an example, we have $S = [S_{AA} S_{AB}; S_{BA} S_{BB}]$, where we have $S_{AB} = S_{BA}$. $S_{ij}$ can be readily computed by [`RPA.form_factor`](@ref) function in the RPA module.

`Smat` should be a `Matrix{<:AbstractArray}`, the matrix indices $i, j$ correspond to the species in the `bcp`. For example, if `species(bcp)` returns `[:A, :B, :C]`, then `Smat[2, 3]` => $S_{BC}$.

Note that current implementation is neither memory nor speed efficient because we don't utilize the fact that $S$ matrix is symmetric.

Each element `Smat` should has the same size of `k2`, the corresponding element in each element of `Smat` is the $S$ value for that k2.
"""
function compute_Smat(bcp::BlockCopolymer, k2::AbstractArray)
    ns = nspecies(bcp)
    Smat = Matrix{typeof(k2)}(undef, ns, ns)
    for i in 1:ns
        for j in 1:ns
            Smat[i, j] = zero(k2)
        end
    end

    return compute_Smat!(Smat, bcp, k2)
end

function compute_Smat!(Smat::AbstractMatrix, bcp::BlockCopolymer, k2::AbstractArray)
    sps = species(bcp)
    ns = nspecies(bcp)
    for i in 1:ns
        for j in 1:ns
            sp1, sp2 = sps[i], sps[j]
            # Approach 1
            Smat[i, j] .+= form_factor(bcp, sp1, sp2, k2)
            # # Approach 2: results in huge number of allocations due to create BlockCopolymerGraph instance.
            # S = Smat[i, j]
            # @inbounds for J in CartesianIndices(k2)
            #     # S_{ij}(k), k is a wave vector.
            #     S[J] += RPA.form_factor(bcp, sp1, sp2, k2[J])
            # end
        end
    end

    return Smat
end

function compute_Smat!(Smat::AbstractMatrix, system::PolymerSystem, k2::AbstractArray)
    sps = species(system)
    ns = nspecies(system)
    sp2id = Dict(Pair.(sps, 1:ns))
    for c in system.components
        if c.molecule isa BlockCopolymer
            sps_bcp = species(c.molecule)
            ns_bcp = nspecies(c.molecule)
            id2sp = Dict(Pair.(1:ns_bcp, sps_bcp))
            Smat_bcp = compute_Smat(c.molecule, k2)
            for i in 1:ns_bcp
                for j in 1:ns_bcp
                    I = sp2id[id2sp[i]]  # map specie index from bcp to system
                    J = sp2id[id2sp[j]]  # map specie index from bcp to system
                    Smat[I,J] .+= c.ϕ / c.α * Smat_bcp[i,j]
                end
            end
        end
    end

    return Smat
end

@doc raw"""
    compute_Smat(system::PolymerSystem, k2::Real)
    compute_Smat!(Smat::AbstractMatrix, system::PolymerSystem, k2::Real)
    compute_Smat(system::PolymerSystem, k2::AbstractArray)
    compute_Smat!(Smat::AbstractMatrix, system::PolymerSystem, k2::AbstractArray)

Compute the $S$ matrix for a polymer system by summing up contributions from all `BlockCopolymer` components. Note the coefficients of `c.ϕ/c.α` should be used to scale the contribution from different `BlockCopolymer` components.
"""
function compute_Smat(system::PolymerSystem, k2::AbstractArray)
    ns = nspecies(system)
    Smat = Matrix{typeof(k2)}(undef, ns, ns)
    for i in 1:ns
        for j in 1:ns
            Smat[i, j] = zero(k2)
        end
    end

    return compute_Smat!(Smat, system, k2)
end

function compute_Smat(bcp::BlockCopolymer, k2::Real)
    sps = species(bcp)
    ns = nspecies(bcp)
    Smat = Matrix{Float64}(undef, ns, ns)
    for i in 1:ns
        for j in 1:ns
            sp1, sp2 = sps[i], sps[j]
            Smat[i, j] = form_factor(bcp, sp1, sp2, k2)
        end
    end

    return Smat
end

function compute_Smat!(Smat::AbstractMatrix, system::PolymerSystem, k2::Real)
    sps = species(system)
    ns = nspecies(system)
    sp2id = Dict(Pair.(sps, 1:ns))
    for c in system.components
        if c.molecule isa BlockCopolymer
            sps_bcp = species(c.molecule)
            ns_bcp = nspecies(c.molecule)
            id2sp = Dict(Pair.(1:ns_bcp, sps_bcp))
            Smat_bcp = c.ϕ / c.α * compute_Smat(c.molecule, k2)
            for i in 1:ns_bcp
                for j in 1:ns_bcp
                    I = sp2id[id2sp[i]]  # map specie index from bcp to system
                    J = sp2id[id2sp[j]]  # map specie index from bcp to system
                    Smat[I,J] += Smat_bcp[i,j]
                end
            end
        end
    end

    return Smat
end

function compute_Smat(system::PolymerSystem, k2::Real)
    ns = nspecies(system)
    Smat = fill(0.0, ns, ns)  # create a ns x ns matrix all initialized to 0.

    return compute_Smat!(Smat, system, k2)
end

@doc raw"""
    compute_Smat_non_interacting(system::PolymerSystem, k2::Real)

Compute the Scattering matrix for a non-interacting polymer system, i.e. where $\chi_{ij}=0$ for all $(i,j)$ pairs. It is the $P$ matrix following the notations in my research notes (2022.4.23).
"""
function compute_Smat_non_interacting(system::PolymerSystem, k2::Real)
    S = compute_Smat(system, k2)
    t = sum(S)
    E = ones(size(S))
    return S .- S*E*S/t
end

@doc raw"""
    compute_Smat_interacting(system::PolymerSystem, k2::Real)

Compute the Scattering matrix for an interacting polymer system. It is the $\tilde{S}$ matrix following the notations in my research notes (2022.4.23).
"""
function compute_Smat_interacting(system::PolymerSystem, k2::Real)
    P = compute_Smat_non_interacting(system, k2)
    V = system.χNmatrix
    return inv(I + P * V) * P
end

function reciprocal_structure_factor(system::PolymerSystem, ::TwoSpeciesSystem, χN::Real, k2::Real)
    S_AA, S_BA, S_AB, S_BB = compute_Smat(system, k2)
    S = S_AA + S_BB + 2S_AB
    W = S_AA .* S_BB - S_AB .* S_AB
    return S/W - 2χN
end

function reciprocal_structure_factor(::PolymerSystem, ::SpecieNumberType, χN, k2)
    return error("Not implemented!")
end

function reciprocal_structure_factor(system::PolymerSystem, χN, k2)
    return reciprocal_structure_factor(system, specie_number_type(system), χN, k2)
end

function structure_factor(system::PolymerSystem, ::TwoSpeciesSystem, χN::Real, k2::Real)
    return 1 / reciprocal_structure_factor(system, χN, k2)
end

function structure_factor(::PolymerSystem, ::SpecieNumberType, χN, k2)
    return error("Not implemented!")
end

function structure_factor(system::PolymerSystem, χN, k2)
    return structure_factor(system, specie_number_type(system), χN, k2)
end

function compute_stability_limit(::BlockCopolymer, ::SpecieNumberType, χN0, k2_max)
    # Use proper error type to throw: not implemented due to the functions exists in the package extension.
    throw(ArgumentError("`compute_stability_limit` is not available. Install Roots.jl and Optim.jl to use this function."))
end

"""
    compute_stability_limit(bcp::BlockCopolymer; χN0=1.0, k2_max=100.0)
    compute_stability_limit(system::PolymerSystem; χN0=1.0, k2_max=100.0)

Compute the stability limit for a block copolymer or a polymer system. Note that only `TwoSpeciesSystem` is supported.
"""
function compute_stability_limit(bcp::BlockCopolymer; χN0=1.0, k2_max=100.0)
    return compute_stability_limit(bcp, specie_number_type(bcp), χN0, k2_max)
end

function compute_stability_limit(::PolymerSystem, ::SpecieNumberType, χN0, k2_max)
    throw(ArgumentError("`compute_stability_limit` is not available. Install Roots.jl and Optim.jl to use this function."))
end

function compute_stability_limit(system::PolymerSystem; χN0=1.0, k2_max=100.0)
    return compute_stability_limit(system, specie_number_type(system), χN0, k2_max)
end

end # module