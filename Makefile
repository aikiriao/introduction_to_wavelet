RM := rm
LATEX := platex
BIBTEX := bibtex
DVIPDFMX := dvipdfmx

# LATEXFLAGS := -interaction=batchmode

TARGET := wavelet_intro_for_seminar.pdf
TARGET_BASENAME := $(basename $(TARGET))

.PHONY: all clean distclean

all: $(TARGET)

clean:
	$(RM) -f *.aux *.log *.dvi *.bcf *.blg *.run.xml *.bbl *.toc *.out *.snm *.nav $(TARGET_BASENAME)-blx.bib

distclean:
	make clean
	$(RM) -f $(TARGET)

rebuild:
	make distclean
	make all

%.pdf: %.dvi
	$(DVIPDFMX) $<

%.dvi: %.tex
	$(LATEX) $(LATEXFLAGS) $<
	# $(BIBTEX) $(basename $<)
	# $(LATEX) $(LATEXFLAGS) $< # 参照を確実に通すため
	# $(LATEX) $(LATEXFLAGS) $< # 参照を確実に通すため
