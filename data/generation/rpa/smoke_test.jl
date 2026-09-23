using Test
include("ExtractedPolyorderRPA.jl")
using .ExtractedPolyorderRPA: RPA

@test RPA.g(0.0, 0.5) == 0.25
@test RPA.h(0.0, 0.5) == 0.5
@test RPA.g(1e-6) ≈ 0.99999966666675

# Symmetric diblock analytical response exercises the unchanged numerical solver.
struct SymmetricDiblockProbe end
RPA.reciprocal_structure_factor(::SymmetricDiblockProbe, chiN, x) =
    2 / (RPA.g(x, 0.5) - RPA.h(x, 0.5)^2) - 2chiN
chiN, wavelength = ExtractedPolyorderRPA.PolyorderRootsExt._compute_stability_limit(
    SymmetricDiblockProbe(), 15.0, 100.0)
@test isapprox(chiN, 10.495; atol=0.002)
@test isapprox(2pi / wavelength, 1.946; atol=0.002)
