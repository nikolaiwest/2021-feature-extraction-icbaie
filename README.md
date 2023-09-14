# Feature Extraction for Time Series Classifications: IEEE-ICBAIE 2021
This repository contains results of following conference publication: 

## Publication
#### Title
Feature extraction for time series classification using univariate descriptive statistics and dynamic time warping in a manufacturing environment

#### Authors
- Nikolai West <sup> [ORCID](https://orcid.org/0000-0002-3657-0211) </sup>
- Thomas Schlegl <sup> [ORCID](https://orcid.org/0000-0003-4094-8085) </sup>
- Jochen Deuse <sup> [ORCID](https://orcid.org/0000-0003-4066-4357) </sup>

#### Abstract 
The decade-long trend toward process automation and end-to-end machine connectivity has fueled an enormous growth of data recorded in the manufacturing industry. Leveraging this potential requires manufacturing companies to extract actionable insights from the data sources. In particular, handling time series data on a large scale requires the use of feature extraction for dimensionality reduction. For this purpose, we propose a new algorithmic approach that uses Dynamic Time Warping to extract maximally discriminative features in a multivariate data set. A benchmark against ‘time series feature extraction based on scalable hypothesis tests’ and state-of-the-art methods, such as InceptionTime, Convolutional Neural Network or ResNet classifier, to evaluate the overall effectiveness. While the proposed algorithm underperforms for an example data set with comparably low dimensionality, scoring 16.67% and 22.80% lower average accuracy than the benchmarks, it achieves competitive results for the real-world application in a manufacturing environment. Here, the average accuracy reaches a delta of just 12.20% and simultaneously reduces computational effort by 97.90%.

#### Conference 
2021 IEEE 2nd International Conference on Big Data, Artificial Intelligence and Internet of Things Engineering (ICBAIE 2021)

#### Status
- Published ([available](https://ieeexplore.ieee.org/document/9389954))

## Usage
The repository contains three separate files that were used for the classification of the gunpoint dataset (GP). Since the manufacturing data could not be provided, GP serves as a demonstration of the feature extraction techniques used throughout the paper. GP is loaded from [pyts](https://pyts.readthedocs.io/en/stable/).

## Contributing
We welcome contributions to this repository. If you have a feature request, bug report, or proposal, please open an issue. If you wish to contribute code, please open a pull request.