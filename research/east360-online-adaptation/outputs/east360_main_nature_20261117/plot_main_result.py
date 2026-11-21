"""Build four independent panels and assemble the 183 × 136 mm main figure."""

import json
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation

from figure_style import DPI, HERE, PT_PER_MM, save_panel
import panel_a
import panel_b
import panel_c
import panel_d


WIDTH_MM, HEIGHT_MM = 183, 136
STEM = HERE / "EAST_main_result_nature"
PANELS = (
    ("a", panel_a, 0, 68),
    ("b", panel_b, 105, 68),
    ("c", panel_c, 0, 0),
    ("d", panel_d, 91.5, 0),
)


def assemble_pdf():
    writer = PdfWriter()
    sheet = writer.add_blank_page(WIDTH_MM * PT_PER_MM, HEIGHT_MM * PT_PER_MM)
    for name, module, x, y in PANELS:
        source = PdfReader(HERE / f"panel_{name}.pdf").pages[0]
        assert abs(float(source.mediabox.width) - module.WIDTH_MM * PT_PER_MM) < 1e-6
        sheet.merge_transformed_page(source, Transformation().translate(x * PT_PER_MM, y * PT_PER_MM))
    sheet.compress_content_streams()
    writer.add_metadata({"/Title": "EAST main experiment: fixed historical replay and extension"})
    with STEM.with_suffix(".pdf").open("wb") as f:
        writer.write(f)


def assemble_svg():
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.Element(f"{{{ns}}}svg", {
        "width": f"{WIDTH_MM}mm", "height": f"{HEIGHT_MM}mm",
        "viewBox": f"0 0 {WIDTH_MM * PT_PER_MM} {HEIGHT_MM * PT_PER_MM}",
        "version": "1.1",
    })
    for name, module, x, y in PANELS:
        panel = ET.parse(HERE / f"panel_{name}.svg").getroot()
        ids = {element.attrib["id"]: f'{name}_{element.attrib["id"]}'
               for element in panel.iter() if "id" in element.attrib}
        for element in panel.iter():
            for key, value in list(element.attrib.items()):
                if key == "id":
                    element.set(key, ids[value])
                else:
                    for old, new in ids.items():
                        value = value.replace(f"url(#{old})", f"url(#{new})")
                        if value == f"#{old}":
                            value = f"#{new}"
                    element.set(key, value)
        panel.set("x", str(x * PT_PER_MM))
        panel.set("y", str((HEIGHT_MM - y - module.HEIGHT_MM) * PT_PER_MM))
        panel.set("width", str(module.WIDTH_MM * PT_PER_MM))
        panel.set("height", str(module.HEIGHT_MM * PT_PER_MM))
        root.append(panel)
    ET.ElementTree(root).write(STEM.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)


def assemble_rasters():
    pixels = lambda mm: round(mm / 25.4 * DPI)
    sheet = Image.new("RGB", (pixels(WIDTH_MM), pixels(HEIGHT_MM)), "white")
    for name, module, x, y in PANELS:
        with Image.open(HERE / f"panel_{name}.png") as im:
            assert abs(im.width - pixels(module.WIDTH_MM)) <= 1
            assert abs(im.height - pixels(module.HEIGHT_MM)) <= 1
            sheet.paste(im.convert("RGB"), (pixels(x), pixels(HEIGHT_MM - y - module.HEIGHT_MM)))
    sheet.save(STEM.with_suffix(".png"), dpi=(DPI, DPI))
    sheet.save(STEM.with_suffix(".tiff"), dpi=(DPI, DPI), compression="tiff_lzw")
    sheet.convert("L").save(HERE / "grayscale_check.png", dpi=(DPI, DPI))


def main():
    placement = []
    for name, module, x, y in PANELS:
        save_panel(module.build(), name)
        placement.append({"panel": name, "x_mm": x, "y_mm": y,
                          "width_mm": module.WIDTH_MM, "height_mm": module.HEIGHT_MM})
    (HERE / "placement.json").write_text(json.dumps(placement, indent=2) + "\n", encoding="utf-8")
    assemble_pdf()
    assemble_svg()
    assemble_rasters()
    print("Built", STEM, "at", WIDTH_MM, "×", HEIGHT_MM, "mm;", DPI, "dpi")


if __name__ == "__main__":
    main()
