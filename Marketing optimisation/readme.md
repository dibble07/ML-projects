# Marketing Optimisation

## Purpose
To build a model to predict whether a marketing call from a loan company would be profitable or not, and subsequently use that model to optimise a marketing call strategy. It is assumed that the cost of making a call is £6 and but the revenue from a successful would yield £80 over the customer's lifetime. The optimisation is to work out which of these customers should be called, and on which day of the week, to maximise profit over the course of a one week marketing call campaign. The constraints of this campaign are:
- Up to 1000 people can be called over the week
- Up to 300 people may be called per day
- Half of people called must be under 45
- 40% of people called must have one of the following jobs: 'blue collar', 'housemaid', 'admin', 'technician'

## Files
| File | Description |
| ----------- | ----------- |
| [requirements.txt](requirements.txt) | List of libraries and version used for python virtual environment |
| [utils.py](utils.py) | Functions for data loading, custom transformers, estimators, calculations and settings etc |
| [prediction.ipynb](prediction.ipynb) | Perform EDA on dataset, build preprocessing pipeline, select and tune ML model |
| [full_predictor.joblib](full_predictor.joblib) | Preprocessing pipeline and estimator as outcome of [Prediction step](prediction.ipynb) |
| [optimisation.ipynb](optimisation.ipynb) | Downselect candidates for marketing call, evaluate randomised strategy, create optimised strategy and compare |

_N.B. some of these files are also present as `.html` files for easy sharing_

## Future improvements
EDA
- [Predictive Power Score](https://github.com/8080labs/ppscore)

Model
- Consider _month_ and _day of week_ as ordinal variable and use symmetry object
- Change decision point of models to maximise custom metric
- Automate fitting of symmetry object
    - Make values private and use decorators for getter and setter methods
- Address multicolinearity in feature set of ML model
    - Could investigate an adaptation of VIF for categorical variables
    - Would improve accuracy of feature importance analysis which may be useful in improving model performance
- Investigate why simpler models seemed to perform better than complex ones

Optimisation
- Predictions are no longer probability of response
- Tune parameters of Genetic Algorithm more thoroughly
- Consider alternative algorithm
    - Simulated annealing
    - Find others that focus on optimising categorical values
- Consider off design optimisation
    - Taguchi array
    - Establish the probably of a call being delayed until the following day and incorporate this into calculation of expected profit