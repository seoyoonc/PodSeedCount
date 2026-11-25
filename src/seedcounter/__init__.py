"""Seed counter: counts seeds inside pods in transmitted-light microscope images."""
__version__ = "2.6.0"

# Version of the automatic analysis itself (pods, contrast, segmentation, confidence features).
# Change it only when a code change alters analysis results; the review then re-analyses images
# that have no edits. Package releases that only add features keep it unchanged. It stays "1.0.0"
# (the version of the analysis it names), so reviews saved with earlier releases remain valid.
ANALYSIS_VERSION = "2.1.0"   # 2.1.0: pod outline from colour distance + edges; 2.0.0: touching pods, ellipses
