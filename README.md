# Geometric Methods for Bayesian Inference and Modeling Discrete Data

This repository contains the introduction part of the doctoral thesis *Geometric Methods for Bayesian Inference and Modeling Discrete Data* by Bernardo Williams.

The thesis introduction is based on four publications, whose full articles are appended to the final dissertation. The preamble chapters in this repository provide the background, notation, methodological context, and summary of contributions that connect the four papers into a single thesis.

## Included Publications

1. **Paper I:** *Riemannian Laplace Approximation with the Fisher Metric*  
   Venue: International Conference on Artificial Intelligence and Statistics (AISTATS 2024)
2. **Paper II:** *Geodesic Slice Sampler for Multimodal Distributions with Strong Curvature*  
   Venue: Conference on Uncertainty in Artificial Intelligence (UAI 2025)
3. **Paper III:** *Simplex-to-Euclidean Bijections for Categorical Flow Matching*  
   Venue: International Conference on Artificial Intelligence and Statistics (AISTATS 2026, accepted/to appear)
4. **Paper IV:** *Simplex-to-Euclidean Bijection for Conjugate and Calibrated Multiclass Gaussian Process Classification*  
   Venue: under review

## Repository Structure

- `main.tex`: main thesis entry point
- `chapters/`: thesis introduction chapters, including the publication list in `chapters/02_publications.tex`
- `bibtex.bib`: bibliography used by the thesis introduction
- `commands.tex`: custom LaTeX commands
- `tktla.cls`: thesis document class
- `figs/`: figures used in the introduction chapters

## Compilation

The thesis can be compiled from the repository root with:

```bash
latexmk --pdf main.tex
```

This is the command currently used for local builds. The project can also be built from Visual Studio Code with a LaTeX extension such as LaTeX Workshop.

### Dependencies

- A LaTeX distribution with `latexmk` installed, such as MacTeX on macOS
- PDFLaTeX support
- BibTeX support for the bibliography
- The LaTeX packages referenced in `main.tex`, provided by a standard TeX distribution

The local `latexmk` version in this environment is:

```text
Latexmk, John Collins, 9 March 2026. Version 4.88
```

## Notes

This repository focuses on the thesis introduction and supporting material that synthesizes the four publications into a unified dissertation on geometric methods for Bayesian inference and modeling discrete data.
