# Project Amygdala: Longitudinal Analysis of Structural Brain Plasticity

This project aims to elucidate the temporal dynamics of amygdala subregions and their interaction with cortical and subcortical areas, utilizing data from the UK Biobank. The analysis focuses on changes in gray matter volume over time and explores the relationship between these changes and a wide range of phenotypic measures.

## Data Preparation

The script requires UK Biobank brain imaging data and phenotypic measures. Ensure that you have access to these datasets and modify the script paths to where your datasets are located.

## Running the Analysis

The main script includes several key steps:

Loading and preprocessing UK Biobank data.
Performing Partial Least Squares Canonical (PLSC) analysis to explore the relationship between amygdala and cortical/subcortical regions' structural changes.
Conducting permutation testing to assess the significance of the observed relationships.
Visualizing the changes in gray matter volume and the results of the PLSC analysis.


The script will generate:

CSV files containing per-participant expression levels for amygdala and brain regions.
Visualizations of median ranked changes in gray matter volume across different age ranges.
Heatmaps showing the relationship between amygdala subregions and cortical/subcortical areas.
Results from permutation testing indicating the significance of the PLSC components.

## Customization

You can modify the analysis by changing the PLSC components, selecting different subsets of the data, or applying the methodology to other brain regions.

## License

This project is open source. Feel free to use and modify the code as needed.

## Acknowledgments

This project utilizes data from the UK Biobank. We thank the UK Biobank and its participants for making this research possible.

## Getting Started

To run the analyses, you will need Python 3 with the following libraries installed:
- NumPy
- Pandas
- NiBabel
- Nilearn
- scikit-learn
- Matplotlib
- Seaborn
- SciPy

You can install these packages via pip:
```bash
pip install numpy pandas nibabel nilearn scikit-learn matplotlib seaborn scipy
