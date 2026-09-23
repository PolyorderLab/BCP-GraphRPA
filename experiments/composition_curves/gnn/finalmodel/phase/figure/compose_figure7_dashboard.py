from base64 import b64encode
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).parent
PANELS = [
    ("(a)", "m1_global_publication", 0, 0),
    ("(b)", "m2_global_publication", 0, 1),
    ("(c)", "btabat_spinodal_publication", 0, 2),
    ("(d)", "m1_comparison_publication", 1, 0),
    ("(e)", "m2_comparison_publication", 1, 1),
    ("(f)", "btabat_spinodal_local_publication", 1, 2),
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


def main():
    panel_w, panel_h, gap, label_h = 1540, 1120, 55, 80
    width, height = panel_w * 3 + gap * 2, panel_h * 2
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 72)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    svg_images = []
    for label, stem, row, col in PANELS:
        keep_legend = row == 0 and col == 2
        x, cell_y = col * (panel_w + gap), row * panel_h
        y = cell_y + label_h
        image = Image.open(HERE / f"{stem}.png").convert("RGB")
        if not keep_legend:
            x0, x1 = ((0.34, 0.80) if row == 0 else (0.42, 0.935))
            ImageDraw.Draw(image).rectangle(
                (int(image.width * x0), int(image.height * 0.065),
                 int(image.width * x1), int(image.height * 0.29)),
                fill="white",
            )
        image.thumbnail((panel_w, panel_h - label_h), Image.Resampling.LANCZOS)
        px, py = x + (panel_w - image.width) // 2, y + (panel_h - label_h - image.height) // 2
        canvas.paste(image, (px, py))
        draw.text((x + 10, cell_y + 2), label, fill="black", font=font)
        source_svg = (HERE / f"{stem}.svg").read_text(encoding="utf-8")
        if not keep_legend:
            source_svg = without_svg_legend(source_svg)
        encoded = b64encode(source_svg.encode()).decode("ascii")
        svg_images.append(
            f'<image x="{x}" y="{y}" width="{panel_w}" height="{panel_h - label_h}" '
            f'preserveAspectRatio="xMidYMid meet" href="data:image/svg+xml;base64,{encoded}"/>'
            f'<text x="{x + 10}" y="{cell_y + 62}" font-family="Arial" font-size="72" '
            f'font-weight="bold">{label}</text>'
        )

    canvas.save(HERE / "figure7a_f_dashboard.png", dpi=(600, 600), optimize=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        + "".join(svg_images) + "</svg>"
    )
    (HERE / "figure7a_f_dashboard.svg").write_text(svg, encoding="utf-8")

    assert canvas.size == (4730, 2240)
    assert all(f">{label}</text>" in svg for label, *_ in PANELS)


if __name__ == "__main__":
    main()
