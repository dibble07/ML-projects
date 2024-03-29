# MRV Emissions Exploration

## Purpose
The European Union requires ships larger than 5000 tons that dock in the EU to declare their emissions to the MRV system. The work presented aims to explore the dataset and investigate any modelling opportunities.

## Files and folders
| File/Folder | Description |
| ----------- | ----------- |
| [data](data) | Raw Excel files as downloaded from [EU-MRV system](https://mrv.emsa.europa.eu/#public/emission-report) |
| [.gitignore](.gitignore) | Files to be ignored in git commits |
| [mrv_emissions_exploration.ipynb](mrv_emissions_exploration.ipynb) | Exploration and modelling notebook |
| [Regulation (EU) 2015-757 of the European Parliament and of the Council.pdf](<Regulation (EU) 2015-757 of the European Parliament and of the Council.pdf>) | Dataset documentation |
| [requirements.txt](requirements.txt) | List of libraries and versions used for python virtual environment |
| [utils.py](utils.py) | Reusable custom functions |

_N.B. some of these files are also present as `.html` files for easy sharing_

## Future improvements
- Automate data download
    1. Scrape web page
        - Identify most recent version of each file to trigger redownload if current version is out of date
        - Extract dates and versions if required for end point URL construction
    1. Download data using Python
        - [Extract end point of web page](https://garycordero1690.medium.com/using-chrome-developers-tools-to-detect-end-points-9b43ad4fdccd)'s download button to hit directly (preferred)
        - Use [selenium to run browser and download file](https://www.browserstack.com/guide/download-file-using-selenium-python) (backup)
- Investigate additional columns that were not loaded from excel for brevity
- Investigate potential insights in columns with lots of missing data
    - Beware of response bias if these fields are not mandatory
    - Understanding a vessel's utilisation rate could be very useful
- Convert to standalone interactive dashboard that can be hosted as web applicable to democratise acess
    - [Voila](https://voila.readthedocs.io/en/stable/) would be well suited
    - Add additional interactive filters (incl/excl ships by year, type, efficiency etc) for end-users to customise
- Replace scatter plot of ship type median speed vs time with bubble plot
    - Use DBScan clustering to aggregate data points into fewer groups for each ship type
    - Plot cluster center and size as bubbles
- Improve regression analysis
    - Overfitting is not a problem so could consider models with higher variance to reduce loss
        - Decision trees
        - Boosted ensemble methods
        - Neural networks
    - Acquire feature around ship size (deadweight, length, volume, displacement etc) not originating from target variable
    - Consider different scoring approach to address range of target values
        - Percentage error

## To do
Address that regression is predicting log 2 of values
- review "#### Evaluate impact of time" down