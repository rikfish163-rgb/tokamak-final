"""Assemble the Seaborn pages into a PDF that can be navigated by shot number."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from pypdf import PdfReader, PdfWriter


HERE = Path(__file__).resolve().parent
ATLAS = HERE / "atlas"
DEMO = {70320: "晚报转及时：12 到 225 ms", 73405: "晚报转及时：29 到 33 ms",
        73743: "额外等 5 点避开误报"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-only", action="store_true", help="reuse the unchanged source-data ZIP")
    args = parser.parse_args()
    rows = json.loads((ATLAS / "index.json").read_text(encoding="utf-8"))
    if len(rows) != 360 or len({r["shot"] for r in rows}) != 360:
        raise ValueError("expected all 360 shots in the fixed replay")
    writer = PdfWriter()
    writer.add_metadata({"/Title": "EAST 360 炮内修正图谱", "/Subject": "回放波形、误差与报警"})
    writer.append(HERE / "EAST360_reader_guide.pdf", import_outline=False)
    writer.add_outline_item("读图前先看：怎样让 EAST 更早发现破裂", 0)
    writer.append(HERE / "EAST360_overview.pdf", import_outline=False)
    writer.add_outline_item("360 炮结果总览", 1)
    writer.append(HERE / "EAST360_readout_360.pdf", import_outline=False)
    writer.add_outline_item("准确率与逐炮误报", 2)
    front_pages = len(writer.pages)
    ids = {r["shot"]: front_pages + r["rank"] - 1 for r in rows}
    examples = writer.add_outline_item("三个案例", front_pages)
    for j in range(6):
        volume = ATLAS / f"EAST360_atlas_{j+1:02d}.pdf"
        reader = PdfReader(volume)
        if len(reader.pages) != 60:
            raise ValueError(f"volume {j+1} has {len(reader.pages)} pages, expected 60")
        writer.append(reader, import_outline=False)
        group = writer.add_outline_item(f"回放第 {60*j+1:03d}–{60*j+60:03d} 炮", front_pages+60*j)
        for row in rows[60*j:60*j+60]:
            rank = row["rank"]
            # Match the extracted page text to the sealed index before adding its bookmark.
            if f"EAST {row['shot']}" not in reader.pages[(rank-1)%60].extract_text():
                raise ValueError(f"unexpected shot page for rank {rank}")
            writer.add_outline_item(f"{rank:03d}｜{row['shot']}", ids[row["shot"]], parent=group)
    for sid, note in DEMO.items():
        writer.add_outline_item(f"{sid}｜{note}", ids[sid], parent=examples)
    audit = json.loads((HERE / "EAST360_评分核对.json").read_text(encoding="utf-8"))
    false_alarms = writer.add_outline_item("第 5 行的 34 炮误报", 2)
    for key, label in (("cold", "准备段"), ("before_write", "第 5 步改动前"), ("after_write", "第 5 步生效后")):
        cases = [r for r in audit["false_alarm_cases"] if r["category"] == key]
        section = writer.add_outline_item(f"{label} · {len(cases)} 炮", ids[cases[0]["shot"]], parent=false_alarms)
        for r in cases:
            sid = r["shot"]
            writer.add_outline_item(f"{sid}｜首次误报 {r['row5_first_s']:.4f} s", ids[sid], parent=section)
    output = HERE / "EAST360_逐炮图谱.pdf"
    temp = output.with_suffix(".pdf.tmp")
    writer.write(temp)
    reread = PdfReader(temp)
    if len(reread.pages) != front_pages + len(rows):
        raise ValueError("wrong page count in assembled atlas")
    for name in DEMO:
        if f"EAST {name}" not in reread.pages[ids[name]].extract_text():
            raise ValueError("wrong bookmark page in assembled atlas")
    temp.replace(output)
    print(f"Verified {len(rows)} named shot pages and a {len(reread.pages)}-page bookmarked PDF.")
    if args.pdf_only:
        return
    # NPZs are already individually compressed; the zip only bundles them.
    dest = HERE / "EAST360_逐炮绘图数据.zip"
    temp = dest.with_suffix(".zip.tmp")
    with ZipFile(temp, "w", compression=ZIP_STORED) as z:
        for path in [ATLAS / "index.csv", ATLAS / "index.json", *sorted((ATLAS / "data").glob("*.npz"))]:
            z.write(path, path.relative_to(ATLAS))
    with ZipFile(temp) as z:
        if len(z.namelist()) != 362 or z.testzip() is not None:
            raise ValueError("incomplete per-shot source bundle")
    temp.replace(dest)
    print(f"Verified all {len(rows)} data files in the ZIP.")


if __name__ == "__main__":
    main()
