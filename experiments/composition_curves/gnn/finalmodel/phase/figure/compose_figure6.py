from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
INPUTS = [
    ROOT / "rpa_ab_series_publication.png",
    ROOT / "rpa_ml_comparison_all_publication.png",
    ROOT / "shi_b2a2_series_publication.png",
]
OUTPUT = ROOT / "figure6_stability_limit_dashboard.png"

panel_width, panel_height = 2400, 1677
header_height, gutter = 160, 24
panels = []

for index, path in enumerate(INPUTS):
    image = Image.open(path).convert("RGB")
    if index < 2:
        # ponytail: erase only the shared top legend; the right panel retains it.
        ImageDraw.Draw(image).rectangle(
            (int(image.width * 0.38), int(image.height * 0.03),
             int(image.width * 0.74), int(image.height * 0.22)),
            fill="white",
        )
    canvas = Image.new("RGB", (image.width, 5858), "white")
    canvas.paste(image, (0, (5858 - image.height) // 2))
    panels.append(canvas.resize((panel_width, panel_height), Image.Resampling.LANCZOS))

dashboard = Image.new(
    "RGB",
    (panel_width * 3 + gutter * 2, header_height + panel_height),
    "white",
)
font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 112)
draw = ImageDraw.Draw(dashboard)
for index, panel in enumerate(panels):
    x = index * (panel_width + gutter)
    draw.text((x + 24, 8), f"({chr(97 + index)})", fill="black", font=font)
    dashboard.paste(panel, (x, header_height))

dashboard.save(OUTPUT, dpi=(600, 600), optimize=True)
assert dashboard.size == (7248, 1837)
print(OUTPUT)
