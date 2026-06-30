# Minimal Sphinx makefile.
# Build the HTML docs with: make html

SPHINXBUILD ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean

help:
	@echo "make html   - build the HTML documentation into $(BUILDDIR)/html"
	@echo "make clean  - remove the build directory"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"
