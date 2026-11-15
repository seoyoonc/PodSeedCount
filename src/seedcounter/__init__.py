"""Seed counter: counts seeds inside pods in transmitted-light microscope images."""
__version__ = "2.4.2"

# Version of the automatic analysis itself (pods, contrast, segmentation, confidence features).
# Change it only when a code change alters analysis results; the review then re-analyses images
# that have no edits. Package releases that only add features keep it unchanged. It stays "1.0.0"
# (the version of the analysis it names), so reviews saved with earlier releases remain valid.
ANALYSIS_VERSION = "1.0.0"
