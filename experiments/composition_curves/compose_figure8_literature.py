# Academic Figure Skill Asset Confirmation (verified against existing manuscript panels)
# (a-c) architecture schematics -> notebook-rendered single panels -> native composition
# (d-i) stability-limit curves -> existing publication panels -> native composition
# ponytail: this compositor only removes duplicated legends and adds panel labels;
# the notebook remains the source of every plotted datum and curve.
from base64 import b64encode
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parents[2] / "results/figures/composition_curves"
SOURCE = HERE / "panel_sources" if (HERE / "panel_sources").is_dir() else HERE
STRUCTURES = [
    ("(a)", "m1_structure_inset", "M1", 0),
    ("(b)", "m2_structure_inset", "M2", 1),
    ("(c)", "m3_structure_inset", "M3", 2),
]
CURVES = [
    ("(d)", "m1_global_publication", 0, 0),
    ("(e)", "m2_global_publication", 0, 1),
    ("(f)", "btabat_spinodal_publication", 0, 2),
    ("(g)", "m1_comparison_publication", 1, 0),
    ("(h)", "m2_comparison_publication", 1, 1),
    ("(i)", "btabat_spinodal_local_publication", 1, 2),
]


def without_svg_legend(svg):
    start = svg.find('<g id="legend_1">')
    if start < 0:
        return svg
    depth = 0
    for match in re.finditer(r"<g\b|</g>", svg[start:]):
        depth += 1 if match.group() == "<g" else -1
        if depth == 0:
            return svg[:start] + svg[start + match.end():]
    raise ValueError("Unclosed SVG legend group")


def embedded_svg(stem, x, y, width, height, keep_legend=True):
    svg = (SOURCE / f"{stem}.svg").read_text(encoding="utf-8")
    if not keep_legend:
        svg = without_svg_legend(svg)
    encoded = b64encode(svg.encode()).decode("ascii")
    return (
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'href="data:image/svg+xml;base64,{encoded}"/>'
    )


def main():
    panel_w, panel_h, gap, label_h = 1540, 1120, 55, 80
    width = panel_w * 3 + gap * 2
    legend_h, structure_h = 130, 850
    height = legend_h + structure_h + panel_h * 2
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 72)
    legend_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 52)
    name_font = ImageFont.truetype(r"C:\Windows\Fonts\ariali.ttf", 60)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    # One centered horizontal legend for all three architecture panels.
    legend_y = 55
    legend_items = [("#1248ff", "A block"), ("#ff7600", "B block")]
    legend_x = width // 2 - 520
    svg_images = []
    for index, (color, text) in enumerate(legend_items):
        x = legend_x + index * 560
        draw.line((x, legend_y, x + 145, legend_y), fill=color, width=18)
        draw.text((x + 180, legend_y - 31), text, fill="black", font=legend_font)
        svg_images.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 145}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="18" stroke-linecap="round"/>'
            f'<text x="{x + 180}" y="{legend_y + 18}" font-family="Arial" '
            f'font-size="52">{text}</text>'
        )

    for label, stem, name, col in STRUCTURES:
        x = col * (panel_w + gap)
        cell_y = legend_h
        y = cell_y + label_h
        image = Image.open(SOURCE / f"{stem}.png").convert("RGB")
        image.thumbnail((panel_w - 130, structure_h - label_h - 90), Image.Resampling.LANCZOS)
        px = x + (panel_w - image.width) // 2
        py = y + (structure_h - label_h - 90 - image.height) // 2
        canvas.paste(image, (px, py))
        draw.text((x + 10, cell_y + 2), label, fill="black", font=font)
        name_box = draw.textbbox((0, 0), name, font=name_font)
        draw.text((x + (panel_w - (name_box[2] - name_box[0])) // 2,
                   cell_y + structure_h - 82), name, fill="black", font=name_font)
        svg_images.append(
            embedded_svg(stem, x + 65, y, panel_w - 130, structure_h - label_h - 90)
        )
        svg_images.append(
            f'<text x="{x + 10}" y="{cell_y + 62}" font-family="Arial" '
            f'font-size="72" font-weight="bold">{label}</text>'
            f'<text x="{x + panel_w / 2}" y="{cell_y + structure_h - 30}" '
            f'font-family="Arial" font-size="60" font-style="italic" '
            f'text-anchor="middle">{name}</text>'
        )

    for label, stem, row, col in CURVES:
        keep_legend = row == 0 and col == 2
        x = col * (panel_w + gap)
        cell_y = legend_h + structure_h + row * panel_h
        y = cell_y + label_h
        image = Image.open(SOURCE / f"{stem}.png").convert("RGB")
        if not keep_legend:
            x0, x1 = ((0.34, 0.80) if row == 0 else (0.42, 0.935))
            ImageDraw.Draw(image).rectangle(
                (int(image.width * x0), int(image.height * 0.065),
                 int(image.width * x1), int(image.height * 0.29)),
                fill="white",
            )
        image.thumbnail((panel_w, panel_h - label_h), Image.Resampling.LANCZOS)
        px = x + (panel_w - image.width) // 2
        py = y + (panel_h - label_h - image.height) // 2
        canvas.paste(image, (px, py))
        draw.text((x + 10, cell_y + 2), label, fill="black", font=font)
        svg_images.append(
            embedded_svg(stem, x, y, panel_w, panel_h - label_h, keep_legend)
        )
        svg_images.append(
            f'<text x="{x + 10}" y="{cell_y + 62}" font-family="Arial" '
            f'font-size="72" font-weight="bold">{label}</text>'
        )

    output = HERE / "figure8_literature_architectures_stability_limits"
    canvas.save(output.with_suffix(".png"), dpi=(600, 600), optimize=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        + "".join(svg_images) + "</svg>"
    )
    output.with_suffix(".svg").write_text(svg, encoding="utf-8")

    assert canvas.size == (4730, 3220)
    assert all(f">({letter})</text>" in svg for letter in "abcdefghi")
    print(output)


if __name__ == "__main__":
    main()
