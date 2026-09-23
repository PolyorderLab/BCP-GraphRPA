# Adapted module imports only; numerical code unchanged. See upstream/LICENSE and upstream/NOTICE.
module PolyorderRootsExt

using Optim: optimize, Brent
using Roots: find_zero
using Polymer: BlockCopolymer, PolymerSystem, TwoSpeciesSystem
using ..RPA: RPA, reciprocal_structure_factor

RPA.compute_stability_limit(bcp::BlockCopolymer, ::TwoSpeciesSystem, χN0, k2_max) = _compute_stability_limit(bcp, χN0, k2_max)

RPA.compute_stability_limit(system::PolymerSystem, ::TwoSpeciesSystem, χN0, k2_max) = _compute_stability_limit(system, χN0, k2_max)

function _compute_stability_limit(bcp_or_system, χN0, k2_max)
    # find x* first. x* is the same for all χN.
    f = x -> reciprocal_structure_factor(bcp_or_system, χN0, x)
    result = optimize(f, 1.0, k2_max, Brent())
    xstar = result.minimizer
    g = y -> reciprocal_structure_factor(bcp_or_system, y, xstar)
    χNstar = find_zero(g, χN0)

    return χNstar, 2π/√xstar
end

end # module
