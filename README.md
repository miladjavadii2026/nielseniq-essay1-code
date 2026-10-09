# Heterogeneous Promotional Effects in NielsenIQ: Code

**Independent Study II, Summer 2026 (Essay 1 foundation)**
Evidence from Cereal and Snacks, with Coffee as a cross-category check.

Milad Javadi, PhD student in Economics, The University of Texas at Dallas
Advisor: Kevin Siqueira

## What this repository contains

Python analysis code for the NielsenIQ phase of Essay 1: fixed-effects estimation (`pyfixest`), promotion-end and long-window event studies, episode duration and inter-purchase hazard models, household fixed effects, double/debiased machine learning, causal forests (`econml`), validation steps, and report QA scripts. Folder paths mirror the working project so script paths stay readable.

Not included: the Stage 1 annual panel build scripts and the scripts that edit report documents.

The research question: which promotions generate incremental demand rather than purchase acceleration or stockpiling, and how do these effects vary across stores, products, categories, and promotion depth?

## What this repository does not contain

**No data and no results.**

- The NielsenIQ Retail Scanner and Consumer Panel data are licensed through the Kilts Center for Marketing at the University of Chicago Booth School of Business. The license does not allow redistribution. To run this code, register with the Kilts Center and obtain the data under your own agreement.
- Result tables, figures, and reports are held back until they complete the Kilts Center pre-publication review.

The `.gitignore` uses a whitelist that admits only `.py` files, so data and output files stay out by default.

## Status

Preliminary research code, not a packaged library. Scripts expect local paths on the author's machine; edit the path constants before running.

## Data disclaimer

Researcher(s)' own analyses calculated (or derived) based in part on data from Nielsen Consumer LLC and marketing databases provided through the NielsenIQ Datasets at the Kilts Center for Marketing Data Center at The University of Chicago Booth School of Business.

The conclusions drawn from the NielsenIQ data are those of the researcher(s) and do not reflect the views of NielsenIQ. NielsenIQ is not responsible for, had no role in, and was not involved in analyzing and preparing the results reported herein.
