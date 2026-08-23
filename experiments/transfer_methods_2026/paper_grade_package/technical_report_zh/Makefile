LATEXMK ?= latexmk
PYTHON ?= python3
OUTDIR := build
PACKAGE_ROOT := ..
FINAL_PDF := $(PACKAGE_ROOT)/TECHNICAL_REPORT_zh.pdf

.PHONY: all sync check clean

all: $(FINAL_PDF)

$(FINAL_PDF): main.tex report_body.tex references.bib $(wildcard figures/*) $(wildcard ../figures/*.pdf)
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(OUTDIR) main.tex
	cp $(OUTDIR)/main.pdf $(FINAL_PDF)

sync:
	$(PYTHON) ../scripts/convert_technical_report_zh_to_latex.py

check: $(FINAL_PDF)
	pdfinfo $(FINAL_PDF)
	pdffonts $(FINAL_PDF)

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR) main.tex
	rm -f $(OUTDIR)/main.bbl $(OUTDIR)/main.run.xml $(OUTDIR)/main.bcf
