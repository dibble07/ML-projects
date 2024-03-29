# Ice Cream Dataset Exploration
_Author: Robert Dibble_

## Purpose
Imagine that you have just joined a well funded Ice Cream start-up as their data scientist. Your task is to find a unique selling point and/or competitive advantage that will ensure their success.

## Steps
Using the [Ice Cream Dataset](https://www.kaggle.com/datasets/tysonpo/ice-cream-dataset) on Kaggle:
1. Perform EDA to understand the market/consumer
1. Identify possible use cases
1. Select a use case and develop POC
1. Provide recommendation with justification

## Files and folders
| File/Folder | Description |
| ----------- | ----------- |
| [.gitignore](.gitignore) | Files to be ignored in git commits |
| [requirements.txt](requirements.txt) | List of libraries and versions used for python virtual environment |
| [data](data) | Raw data files |
| [ice_cream_dataset_exploration.ipynb](ice_cream_dataset_exploration.ipynb) |  |

_N.B. some of these files are also present as `.html` files for easy sharing_

## Future improvements
- Improve tokenisation
    - Americanisms
    - Plurals
    - Multi-component ice creams - denoted by 'component1: ingredient, ingredient. component2: ingredient, ingredient'
    - Optional prefix of "may" in allergen warnings
- Weight rating average by number of reviews
    - Extend (Laplace's rule of succession)[https://en.wikipedia.org/wiki/Rule_of_succession] to the 0-5 star rating system
- Identify pairs of ingredients that are important
    - Aggregate shap values across the presence of multiple ingredients
- Calculate uncertainty
    - Instead of aggregating across ingredient presence, regress to obtain confidence intervals
- Unit testing of custom functions
    - String processing functions