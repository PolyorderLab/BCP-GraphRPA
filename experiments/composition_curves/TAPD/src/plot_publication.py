"""
Publication-quality phase diagram plots using MakiePublication.jl
"""

import subprocess
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional


def save_data_for_julia(rpa_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       ml_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
                       output_path: str):
    """Save data as JSON for Julia to read."""
    data = {}
    for topo in ['AB', 'ABA', 'ABAB', 'ABABA', 'AB2', 'AB3', 'M1', 'M2']:
        if topo in rpa_data:
            fA, chiN = rpa_data[topo]
            data[topo] = {
                'rpa': {'fA': fA.tolist(), 'chiN': chiN.tolist()}
            }
        if ml_data is not None and topo in ml_data:
            fA, chiN = ml_data[topo]
            data[topo]['ml'] = {'fA': fA.tolist(), 'chiN': chiN.tolist()}

    with open(output_path, 'w') as f:
        json.dump(data, f)


def plot_with_makie(data_path: str, output_path: str, publication: bool = False):
    """Generate combined phase diagram using MakiePublication.jl"""
    grid_option = "false" if publication else "true"
    lw_rpa = 4.0 if publication else 2.0
    lw_ml = 2.5 if publication else 1.5
    fontsize_base = 32 if publication else 12
    ms = 12 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie

data_path = "{data_path}"
output_path = "{output_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "AB" => "#2E86AB",
    "ABA" => "#A23B72",
    "ABAB" => "#F18F01",
    "ABABA" => "#C73E1D",
    "AB2" => "#3A7D44",
    "AB3" => "#9B59B6",
    "M1" => "#E67E22",
    "M2" => "#1ABC9C"
)

topologies = ["AB", "ABA", "ABAB", "ABABA"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
        fig = Figure(figure_padding=(5, 15, 2, 2), size=(fw, round(fw * 0.75)))
    else
        fig = Figure(figure_padding=(5, 15, 2, 2), size=(1200, 900))
    end

    ax = Axis(fig[1, 1],
        xlabel = L"$f_A$",
        ylabel = L"($\\chi N)^*$",
        xlabelsize = {fontsize_base},
        ylabelsize = {fontsize_base},
        xticklabelsize = {fontsize_base},
        yticklabelsize = {fontsize_base},
        xgridwidth = 1.5,
        ygridwidth = 1.5,
        titlesize = {fontsize_base + 4},
        titlefont = :bold,
        xlabelpadding = 15,
        ylabelpadding = 15
    )

    ax.xgridvisible = {grid_option}
    ax.ygridvisible = {grid_option}
    ax.yticks = [0, 50, 100, 150, 200]
    ax.xticks = [0.2, 0.4, 0.6, 0.8, 1.0]

    for topo in topologies
        if haskey(data, topo) && haskey(data[topo], "rpa")
            c = colors[topo]
            rpa_data = data[topo]["rpa"]
            lines!(ax, rpa_data["fA"], rpa_data["chiN"];
                color = c, linewidth = {lw_rpa}, linestyle = :solid, label = "$topo RPA"
            )
        end

        if haskey(data, topo) && haskey(data[topo], "ml")
            c = colors[topo]
            ml_data = data[topo]["ml"]
            scatter!(ax, ml_data["fA"], ml_data["chiN"];
                color = c, markersize = {ms}, strokecolor = c, strokewidth = 1, label = "$topo ML"
            )
        end
    end

    Legend(fig[1, 2], ax, valign=:top, padding=(5, 5, 5, 5), labelsize={fontsize_base}, rowgap=15)
    xlims!(ax, 0.0, 1.0)
    ylims!(ax, 0.0, 210.0)

    save(output_path, fig)
    println("Saved: $output_path")
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_grid_with_makie(data_path: str, output_path: str, publication: bool = False):
    """Generate 2x2 grid plot using MakiePublication.jl"""
    grid_option = "false" if publication else "true"
    lw_rpa = 6.0 if publication else 2.5
    lw_ml = 5.0 if publication else 2.0
    fontsize_base = 48 if publication else 12
    ms = 14 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie

data_path = "{data_path}"
output_path = "{output_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "AB" => "#2E86AB",
    "ABA" => "#A23B72",
    "ABAB" => "#F18F01",
    "ABABA" => "#C73E1D",
    "AB2" => "#3A7D44",
    "AB3" => "#9B59B6",
    "M1" => "#E67E22",
    "M2" => "#1ABC9C"
)

topologies = ["AB", "ABA", "ABAB", "ABABA"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
        fig = Figure(figure_padding=(20, 40, 15, 15), size=(fw, round(fw * 0.75)))
    else
        fw = 2000.0
        fig = Figure(figure_padding=(20, 40, 15, 15), size=(2600, 1950))
    end

    for (idx, topo) in enumerate(topologies)
        row = div(idx-1, 2) + 1
        col = mod(idx-1, 2) + 1
        ax = Axis(fig[row, col],
            xlabel = L"$f_A$",
            ylabel = L"($\\chi N)^*$",
            xlabelsize = {fontsize_base},
            ylabelsize = {fontsize_base},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            title = topo,
            titlesize = {fontsize_base + 6},
            titlecolor = :black,
            titlefont = :bold,
            xlabelpadding = 20,
            ylabelpadding = 20
        )

        ax.xgridvisible = {grid_option}
        ax.ygridvisible = {grid_option}
        ax.yticks = [0, 100, 200, 300, 400, 500]

        if haskey(data, topo) && haskey(data[topo], "rpa")
            c = colors[topo]
            rpa_data = data[topo]["rpa"]
            lines!(ax, rpa_data["fA"], rpa_data["chiN"];
                color = c, linewidth = {lw_rpa}, linestyle = :solid, label = "$topo RPA"
            )
        end

        if haskey(data, topo) && haskey(data[topo], "ml")
            c = colors[topo]
            ml_data = data[topo]["ml"]
            scatter!(ax, ml_data["fA"], ml_data["chiN"];
                color = c, markersize = {ms}, strokecolor = c, strokewidth = 1, label = "$topo ML"
            )
        end

        xlims!(ax, 0.0, 1.0)
        ylims!(ax, 0.0, 550.0)
    end

    # Force equal row heights (shorter) and column widths (wider)
    colsize!(fig.layout, 1, Fixed(1000))
    colsize!(fig.layout, 2, Fixed(1000))
    rowsize!(fig.layout, 1, Fixed(550))
    rowsize!(fig.layout, 2, Fixed(550))

    # Legend with all topology labels in 2 columns (RPA and ML)
    elements = []
    labels = []
    for topo in topologies
        push!(elements, LineElement(color = colors[topo], linestyle = :solid, linewidth = {lw_rpa}))
        push!(labels, "$topo RPA")
        push!(elements, MarkerElement(color = colors[topo], marker = :circle, markersize = {ms}))
        push!(labels, "$topo ML")
    end
    Legend(fig[1, 3], elements, labels, valign=:top, padding=(10, 10, 10, 10), labelsize={fontsize_base}, rowgap=15, orientation=:vertical, ncols=2)

    save(output_path, fig)
    println("Saved: $output_path")
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_error_analysis_with_makie(data_path: str, output_dir: str, publication: bool = True):
    """Generate error analysis plots using MakiePublication.jl"""
    lw = 3.0 if publication else 2.0
    fontsize_base = 16 if publication else 12
    markersize = 10 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie
using Statistics

data_path = "{data_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "AB" => "#2E86AB",
    "ABA" => "#A23B72",
    "ABAB" => "#F18F01",
    "ABABA" => "#C73E1D",
    "AB2" => "#3A7D44",
    "AB3" => "#9B59B6",
    "M1" => "#E67E22",
    "M2" => "#1ABC9C"
)

topologies = ["AB", "ABA", "ABAB", "ABABA"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
    else
        fw = 1200.0
    end

    # Error vs fA plots (first row)
    for (idx, topo) in enumerate(topologies)
        fig = Figure(figure_padding=(5, 5, 2, 2), resolution=(fw * 0.45, fw * 0.4))

        ax = Axis(fig[1, 1],
            xlabel = L"$f_A$",
            ylabel = "Relative Error (%)",
            xlabelsize = {fontsize_base + 2},
            ylabelsize = {fontsize_base + 2},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            title = "$topo: Prediction Error",
            titlesize = {fontsize_base + 2},
            titlefont = :bold
        )

        ax.xgridvisible = true
        ax.ygridvisible = true

        if haskey(data, topo) && haskey(data[topo], "rpa") && haskey(data[topo], "ml")
            rpa_chiN = data[topo]["rpa"]["chiN"]
            ml_chiN = data[topo]["ml"]["chiN"]
            fA = data[topo]["rpa"]["fA"]

            # Calculate relative error
            error = Float64[]
            for i in 1:length(rpa_chiN)
                if rpa_chiN[i] != 0
                    push!(error, (ml_chiN[i] - rpa_chiN[i]) / rpa_chiN[i] * 100)
                else
                    push!(error, 0.0)
                end
            end

            c = colors[topo]
            scatter!(ax, fA, error;
                color = c, markersize = {markersize}, strokecolor = c, strokewidth = 1,
                label = "Error"
            )
            lines!(ax, fA, error, color = c, linewidth = {lw})

            hlines!(ax, 0.0; color=:gray, linewidth=1.5, linestyle=:dash)
        end

        xlims!(ax, 0.1, 0.9)
        save(joinpath("{output_dir}", "error_analysis_$topo.png"), fig)
        println("Saved: error_analysis_$topo.png")
    end

    # Parity plots (second row)
    for (idx, topo) in enumerate(topologies)
        fig = Figure(figure_padding=(5, 5, 2, 2), resolution=(fw * 0.45, fw * 0.45))

        ax = Axis(fig[1, 1],
            xlabel = L"$\chi N^*_{{RPA}}$",
            ylabel = L"$\chi N^*_{{ML}}$",
            xlabelsize = {fontsize_base + 2},
            ylabelsize = {fontsize_base + 2},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            title = "$topo: Parity Plot",
            titlesize = {fontsize_base + 2},
            titlefont = :bold,
            xscale = log10,
            yscale = log10
        )

        ax.xgridvisible = true
        ax.ygridvisible = true

        if haskey(data, topo) && haskey(data[topo], "rpa") && haskey(data[topo], "ml")
            rpa_chiN = data[topo]["rpa"]["chiN"]
            ml_chiN = data[topo]["ml"]["chiN"]

            c = colors[topo]
            scatter!(ax, rpa_chiN, ml_chiN;
                color = c, markersize = {markersize}, strokecolor = c, strokewidth = 1,
                label = "Predictions"
            )

            # Diagonal line
            min_val = min(minimum(rpa_chiN), minimum(ml_chiN))
            max_val = max(maximum(rpa_chiN), maximum(ml_chiN))
            lines!(ax, [min_val, max_val], [min_val, max_val];
                color = :gray, linewidth = 2, linestyle = :dash, label = "Perfect"
            )

            # Calculate metrics
            errors = abs.(ml_chiN - rpa_chiN) ./ rpa_chiN .* 100
            mape = mean(errors)
            text!(ax, 0.05, 0.95, text = "MAPE: $(round(mape, digits=2))%", position = (0.05, 0.95),
                align = (:left, :top), fontsize = {fontsize_base})
        end

        Legend(fig[1, 2], ax, valign=:top, padding=(5, 5, 5, 5))
        save(joinpath("{output_dir}", "parity_plot_$topo.png"), fig)
        println("Saved: parity_plot_$topo.png")
    end
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_ab_series_with_makie(data_path: str, output_path: str, publication: bool = True):
    """Generate AB, AB2, AB3 series plot using MakiePublication.jl"""
    lw_rpa = 3.0 if publication else 2.5
    lw_ml = 2.5 if publication else 2.0
    fontsize_base = 32 if publication else 12

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie

data_path = "{data_path}"
output_path = "{output_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "AB" => "#2E86AB",
    "AB2" => "#3A7D44",
    "AB3" => "#9B59B6"
)

topologies = ["AB", "AB2", "AB3"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
        fig = Figure(figure_padding=(5, 15, 2, 2), size=(fw, round(fw * 0.75)))
    else
        fig = Figure(figure_padding=(5, 15, 2, 2), size=(1200, 900))
    end

    ax = Axis(fig[1, 1],
        xlabel = L"$f_A$",
        ylabel = L"($\\chi N)^*$",
        xlabelsize = {fontsize_base},
        ylabelsize = {fontsize_base},
        xticklabelsize = {fontsize_base},
        yticklabelsize = {fontsize_base},
        xgridwidth = 1.5,
        ygridwidth = 1.5,
        titlesize = {fontsize_base + 4},
        titlefont = :bold,
        xlabelpadding = 15,
        ylabelpadding = 15
    )

    ax.xgridvisible = false
    ax.ygridvisible = false
    ax.yticks = [0, 50, 100, 150, 200]
    ax.xticks = [0.2, 0.4, 0.6, 0.8, 1.0]

    for topo in topologies
        if haskey(data, topo) && haskey(data[topo], "rpa")
            c = colors[topo]
            rpa_data = data[topo]["rpa"]
            lines!(ax, rpa_data["fA"], rpa_data["chiN"];
                color = c, linewidth = {lw_rpa}, linestyle = :solid, label = "$topo RPA"
            )
        end

        if haskey(data, topo) && haskey(data[topo], "ml")
            c = colors[topo]
            ml_data = data[topo]["ml"]
            scatter!(ax, ml_data["fA"], ml_data["chiN"];
                color = c, markersize = {12}, strokecolor = c, strokewidth = 1, label = "$topo ML"
            )
        end
    end

    Legend(fig[1, 2], ax, valign=:top, padding=(5, 5, 5, 5), labelsize={fontsize_base}, rowgap=15)
    xlims!(ax, 0.0, 1.0)
    ylims!(ax, 0.0, 210.0)

    save(output_path, fig)
    println("Saved: $output_path")
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_ab_series_grid_with_makie(data_path: str, output_path: str, publication: bool = True):
    """Generate 3x1 grid plot for AB series (AB, AB2, AB3) using MakiePublication.jl"""
    grid_option = "false" if publication else "true"
    lw_rpa = 6.0 if publication else 2.5
    lw_ml = 5.0 if publication else 2.0
    fontsize_base = 48 if publication else 12
    ms = 14 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie

data_path = "{data_path}"
output_path = "{output_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "AB" => "#2E86AB",
    "AB2" => "#3A7D44",
    "AB3" => "#9B59B6"
)

topologies = ["AB", "AB2", "AB3"]

with_theme(theme_aps()) do
    fw = 2400.0
    fig = Figure(figure_padding=(20, 30, 15, 15), size=(fw, fw * 0.35))

    for (idx, topo) in enumerate(topologies)
        ax = Axis(fig[1, idx],
            xlabel = L"$f_A$",
            ylabel = L"($\\chi N)^*$",
            xlabelsize = {fontsize_base},
            ylabelsize = {fontsize_base},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            title = topo,
            titlesize = {fontsize_base + 6},
            titlecolor = :black,
            titlefont = :bold,
            xlabelpadding = 20,
            ylabelpadding = 20
        )

        ax.xgridvisible = {grid_option}
        ax.ygridvisible = {grid_option}
        ax.yticks = [0, 100, 200, 300, 400, 500]

        if haskey(data, topo) && haskey(data[topo], "rpa")
            c = colors[topo]
            rpa_data = data[topo]["rpa"]
            lines!(ax, rpa_data["fA"], rpa_data["chiN"];
                color = c, linewidth = {lw_rpa}, linestyle = :solid, label = "RPA"
            )
        end

        if haskey(data, topo) && haskey(data[topo], "ml")
            c = colors[topo]
            ml_data = data[topo]["ml"]
            scatter!(ax, ml_data["fA"], ml_data["chiN"];
                color = c, markersize = {ms}, strokecolor = c, strokewidth = 1, label = "ML"
            )
        end

        xlims!(ax, 0.0, 1.0)
        ylims!(ax, 0.0, 550.0)
    end

    # Legend
    elements = [
        LineElement(color = :gray, linestyle = :solid, linewidth = {lw_rpa}),
        MarkerElement(color = :gray, marker = :circle, markersize = {ms})
    ]
    labels = ["RPA", "ML"]
    Legend(fig[1, 4], elements, labels, valign=:center, padding=(10, 10, 10, 10), labelsize={fontsize_base}, rowgap=15)

    save(output_path, fig)
    println("Saved: $output_path")
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_combined_analysis_with_makie(data_path: str, output_dir: str, publication: bool = True):
    """Generate combined parity plots and error analysis plots (2x1 layout) using MakiePublication.jl"""
    lw = 3.0 if publication else 2.0
    fontsize_base = 20 if publication else 12
    markersize = 12 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie
using Statistics

data_path = "{data_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "AB" => "#2E86AB",
    "ABA" => "#A23B72",
    "ABAB" => "#F18F01",
    "ABABA" => "#C73E1D",
    "AB2" => "#3A7D44",
    "AB3" => "#9B59B6",
    "M1" => "#E67E22",
    "M2" => "#1ABC9C"
)

topologies = ["AB", "ABA", "ABAB", "ABABA"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
    else
        fw = 1600.0
    end

    for (idx, topo) in enumerate(topologies)
        fig = Figure(figure_padding=(5, 5, 2, 2), size=(fw, fw * 0.45))

        # Left: Parity Plot
        ax1 = Axis(fig[1, 1],
            xlabel = L"$\chi N^*_{{RPA}}$",
            ylabel = L"$\chi N^*_{{ML}}$",
            xlabelsize = {fontsize_base},
            ylabelsize = {fontsize_base},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            titlesize = {fontsize_base + 2},
            titlefont = :bold,
            xscale = log10,
            yscale = log10
        )

        ax1.xgridvisible = true
        ax1.ygridvisible = true

        # Right: Error Plot
        ax2 = Axis(fig[1, 2],
            xlabel = L"$f_A$",
            ylabel = "Relative Error (%)",
            xlabelsize = {fontsize_base},
            ylabelsize = {fontsize_base},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            titlesize = {fontsize_base + 2},
            titlefont = :bold
        )

        ax2.xgridvisible = true
        ax2.ygridvisible = true

        if haskey(data, topo) && haskey(data[topo], "rpa") && haskey(data[topo], "ml")
            rpa_chiN = data[topo]["rpa"]["chiN"]
            ml_chiN = data[topo]["ml"]["chiN"]
            fA = data[topo]["rpa"]["fA"]
            c = colors[topo]

            # Parity plot (left)
            scatter!(ax1, rpa_chiN, ml_chiN;
                color = c, markersize = {markersize}, strokecolor = c, strokewidth = 1
            )

            # Diagonal line with x_min = 10^1
            min_val = max(10.0, min(minimum(rpa_chiN), minimum(ml_chiN)))
            max_val = max(maximum(rpa_chiN), maximum(ml_chiN))
            lines!(ax1, [min_val, max_val], [min_val, max_val];
                color = :gray, linewidth = 2, linestyle = :dash
            )

            # Set axis limits for parity plot
            xlims!(ax1, 10.0, max_val * 1.1)
            ylims!(ax1, 10.0, max_val * 1.1)

            # Calculate MAPE
            errors = abs.(ml_chiN - rpa_chiN) ./ rpa_chiN .* 100
            mape = mean(errors)
            text!(ax1, 0.05, 0.95, text = "MAPE: $(round(mape, digits=2))%", position = (0.05, 0.95),
                align = (:left, :top), fontsize = {fontsize_base - 2}, space = :relative)

            # Error plot (right)
            error = Float64[]
            for i in 1:length(rpa_chiN)
                if rpa_chiN[i] != 0
                    push!(error, (ml_chiN[i] - rpa_chiN[i]) / rpa_chiN[i] * 100)
                else
                    push!(error, 0.0)
                end
            end

            scatter!(ax2, fA, error;
                color = c, markersize = {markersize}, strokecolor = c, strokewidth = 1
            )
            lines!(ax2, fA, error, color = c, linewidth = {lw})
            hlines!(ax2, 0.0; color=:gray, linewidth=1.5, linestyle=:dash)

            xlims!(ax2, 0.1, 0.9)
        end

        save(joinpath("{output_dir}", "combined_analysis_$topo.png"), fig)
        println("Saved: combined_analysis_$topo.png")
    end
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_m1_m2_global_with_makie(data_path: str, output_path: str, publication: bool = True):
    """Generate M1 and M2 global comparison plot using MakiePublication.jl."""
    lw = 4.0 if publication else 2.0
    fontsize_base = 28 if publication else 12
    ms = 10 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie

data_path = "{data_path}"
output_path = "{output_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "M1" => "#E67E22",
    "M2" => "#1ABC9C"
)

topologies = ["M1", "M2"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
        fig = Figure(figure_padding=(5, 10, 2, 2), size=(fw, round(fw * 0.65)))
    else
        fig = Figure(figure_padding=(10, 10, 5, 5), size=(1600, 600))
    end

    for (idx, topo) in enumerate(topologies)
        ax = Axis(fig[1, idx],
            xlabel = L"$f_A$",
            ylabel = L"($\\chi N)^*$",
            xlabelsize = {fontsize_base},
            ylabelsize = {fontsize_base},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            titlesize = {fontsize_base + 4},
            titlefont = :bold,
            title = topo,
            xlabelpadding = 15,
            ylabelpadding = 15
        )

        ax.xgridvisible = false
        ax.ygridvisible = false
        ax.yticks = [0, 50, 100, 150, 200]
        ax.xticks = [0.1, 0.3, 0.5, 0.7, 0.9]

        c = colors[topo]

        if haskey(data, topo) && haskey(data[topo], "rpa")
            rpa_data = data[topo]["rpa"]
            lines!(ax, rpa_data["fA"], rpa_data["chiN"];
                color = c, linewidth = {lw}, linestyle = :solid, label = "RPA"
            )
        end

        if haskey(data, topo) && haskey(data[topo], "ml")
            ml_data = data[topo]["ml"]
            scatter!(ax, ml_data["fA"], ml_data["chiN"];
                color = c, markersize = {ms}, strokecolor = c, strokewidth = 1.5, label = "ML"
            )
        end

        xlims!(ax, 0.0, 1.0)
        ylims!(ax, 0.0, 210.0)
    end

    Legend(fig[1, 3], [
        LineElement(color = :black, linestyle = :solid, linewidth = {lw}),
        MarkerElement(color = :black, marker = :circle, markersize = {ms})
    ], ["RPA", "ML"],
    halign = 1.0, valign = 1.0,
    padding = (5, 5, 5, 5),
    labelsize = {fontsize_base - 4},
    rowgap = 5)

    save(output_path, fig)
    println("Saved: $output_path")
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def plot_m1_m2_with_makie(data_path: str, output_path: str, publication: bool = True):
    """Generate M1 and M2 ODT comparison plot (2 subplots side by side) 
    using MakiePublication.jl.
    
    Left subplot: M1, Right subplot: M2
    - No grid lines
    - x-axis fA: 0.11 to 0.19
    - y-axis chiN: 50 to 150 (150 fully visible)
    - RPA: solid lines, ML: scatter points
    """
    lw = 4.0 if publication else 2.0
    fontsize_base = 28 if publication else 12
    ms = 10 if publication else 6

    julia_script = f'''
using Pkg
Pkg.activate("/home/zxy/project/jl")
using MakiePublication
using CairoMakie

data_path = "{data_path}"
output_path = "{output_path}"

using JSON
data = JSON.parse(read(data_path, String))

colors = Dict(
    "M1" => "#E67E22",
    "M2" => "#1ABC9C"
)

topologies = ["M1", "M2"]

with_theme(theme_aps()) do
    if haskey(ENV, "FIGURE_WIDTH")
        fw = parse(Float64, ENV["FIGURE_WIDTH"])
        fig = Figure(figure_padding=(5, 10, 2, 2), size=(fw, round(fw * 0.65)))
    else
        # 每个子图 4:1 (宽:高)
        fig = Figure(figure_padding=(10, 10, 5, 5), size=(1600, 600))
    end

    for (idx, topo) in enumerate(topologies)
        ax = Axis(fig[1, idx],
            xlabel = L"$f_A$",
            ylabel = L"($\\chi N)^*$",
            xlabelsize = {fontsize_base},
            ylabelsize = {fontsize_base},
            xticklabelsize = {fontsize_base},
            yticklabelsize = {fontsize_base},
            titlesize = {fontsize_base + 4},
            titlefont = :bold,
            title = topo,
            xlabelpadding = 15,
            ylabelpadding = 15
        )

        ax.xgridvisible = false
        ax.ygridvisible = false
        ax.yticks = [50, 75, 100, 125, 150]

        c = colors[topo]

        # RPA ODT: solid line
        if haskey(data, topo) && haskey(data[topo], "rpa")
            rpa_data = data[topo]["rpa"]
            lines!(ax, rpa_data["fA"], rpa_data["chiN"];
                color = c, linewidth = {lw}, linestyle = :solid, label = "RPA"
            )
        end

        # ML ODT: scatter points
        if haskey(data, topo) && haskey(data[topo], "ml")
            ml_data = data[topo]["ml"]
            scatter!(ax, ml_data["fA"], ml_data["chiN"];
                color = c, markersize = {ms}, strokecolor = c, strokewidth = 1.5, label = "ML"
            )
        end

        xlims!(ax, 0.11, 0.19)
        ylims!(ax, 50.0, 155.0)
    end

    # Legend at top-right of figure
    Legend(fig[1, 3], [
        LineElement(color = :black, linestyle = :solid, linewidth = {lw}),
        MarkerElement(color = :black, marker = :circle, markersize = {ms})
    ], ["RPA", "ML"],
    halign = 1.0, valign = 1.0,
    padding = (5, 5, 5, 5),
    labelsize = {fontsize_base - 4},
    rowgap = 5)

    save(output_path, fig)
    println("Saved: $output_path")
end
'''

    result = subprocess.run(['julia', '-e', julia_script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Julia stdout: {result.stdout}")
        print(f"Julia stderr: {result.stderr}")


def generate_publication_plots(rpa_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              ml_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
                              output_dir: str):
    """Generate all publication-quality plots using MakiePublication.jl"""
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "plot_data.json")

    save_data_for_julia(rpa_data, ml_data, data_path)

    print("Generating publication-quality plots with MakiePublication.jl...")

    # Combined plot (publication style - no grid, large fonts, thick lines)
    plot_with_makie(data_path, os.path.join(output_dir, "rpa_ml_comparison_all_publication.png"), publication=True)

    # Grid plot (publication style)
    plot_grid_with_makie(data_path, os.path.join(output_dir, "rpa_ml_comparison_grid_publication.png"), publication=True)

    # AB series plot (AB, AB2, AB3)
    plot_ab_series_with_makie(data_path, os.path.join(output_dir, "rpa_ab_series_publication.png"), publication=True)

    # AB series grid plot (AB, AB2, AB3 in 1x3 grid)
    plot_ab_series_grid_with_makie(data_path, os.path.join(output_dir, "rpa_ab_series_grid_publication.png"), publication=True)

    # Error analysis plots (combined with parity plots in 2x1 layout)
    plot_combined_analysis_with_makie(data_path, output_dir, publication=True)

    # M1 and M2 comparison plot
    plot_m1_m2_with_makie(data_path, os.path.join(output_dir, "m1_m2_comparison_publication.png"), publication=True)

    # M1 and M2 global plot
    plot_m1_m2_global_with_makie(data_path, os.path.join(output_dir, "m1_m2_global_publication.png"), publication=True)

    print(f"All plots saved to: {output_dir}")
