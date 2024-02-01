# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
random.seed(42)
import scipy.stats as stats
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from statsmodels.stats import proportion
import tensorflow as tf
import unittest

# load and split required data
def get_data():

    # load dataset - specify dtype to match specification and save computational memory
    raw_df = pd.read_csv(
        'dataset.csv',
        dtype = {
            'age': int,
            'job': "category",
            'marital_status': "category",
            'education': "category",
            'default': "category",
            'housing': "category",
            'loan': "category",
            'contact': "category",
            'month': "category",
            'day_of_week': "category",
            'days_since_last_contact': int,
            'n_previous_contacts': int,
            'previous_outcome': "category",
            'success': bool
        }
    )

    # define ordinal category orders
    ordinality_dict = {
        'education': ['unknown', 'illiterate', '4y', '6y', '9y', 'high_school', 'professional_course', 'university_degree'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    }

    # split train and test sets
    # train_df = raw_df.loc[raw_df['month'].isin(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep'])]
    # test_df = raw_df.loc[raw_df['month'].isin(['oct', 'nov', 'dec'])]
    train_valid_df, test_df = train_test_split(raw_df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.1, random_state=42)
    assert not set(test_df.index).intersection(set(valid_df.index)).intersection(set(train_df.index)) # check no overlap between sets
    
    # extract variables and target
    y_train_df, y_valid_df, y_test_df = train_df['success'], valid_df['success'], test_df['success']
    X_train_df, X_valid_df, X_test_df = train_df.drop(columns=['success']), valid_df.drop(columns=['success']), test_df.drop(columns=['success'])
    
    return y_train_df, y_valid_df, y_test_df, X_train_df, X_valid_df, X_test_df, ordinality_dict

# function to calculate relevant correlation values for categorical target type
def corr_fun(X, y):

    # assess data type
    is_num = pd.api.types.is_numeric_dtype(X)

    # calculate correlation statistics and write title string
    if is_num:
        corr_val, p_val = stats.pointbiserialr(X, y) # pointbiserialr is a correlation coefficient for numerical-categorical relationships
        title = f'pointbiserialr correlation={corr_val:.3f}\npointbiserialr p-value={p_val:.3f}'
    else:
        _, p_val, _, _ = stats.chi2_contingency(pd.crosstab(y, X)) # chi squared is a test for categorical-categorical relationships
        corr_val = np.NaN
        title = f'\u03A7 squared p-value={p_val:.3f}'

    # create dataframe of output
    df = pd.Series({'numerical': is_num, 'correlation_coefficient': corr_val, 'p-value': p_val})
    df = df.to_frame(X.name).transpose().astype({'numerical':'bool', 'correlation_coefficient':'float', 'p-value':'float'})
    
    return df, title

# function to plot distribution of variable and relationship to target
def eda_plot(X, y, vars_df, ord, ax1, ax2):

    # calculate correlations
    vars_new_df, title_str = corr_fun(X, y)
    vars_df = pd.concat([vars_df, vars_new_df])

    # group by column value
    col_df_group = pd.concat([X, y], axis=1).groupby(X.name).agg({'count','mean'})
    col_df_group.columns = col_df_group.columns.droplevel()

    # return ordinality if appropriate
    if X.name in ord.keys():
        assert set(col_df_group.index).issubset(set(ord[X.name])) # check values in data match those in prescribed ordinality order
        req_inds = [x for x in ord[X.name] if x in col_df_group.index]
        col_df_group = col_df_group.loc[req_inds]

    # plot occurance of feature value and retention confidence interval
    xlabels = col_df_group.index
    ax1.bar(xlabels, col_df_group['count'], color='darkgray')
    err_below, err_above = proportion.proportion_confint(count=col_df_group['count']*col_df_group['mean'], nobs=col_df_group['count'])
    err_arr = np.concatenate(((col_df_group['mean']-err_below).values.reshape(1,-1), (err_above-col_df_group['mean']).values.reshape(1,-1)))
    ax2.errorbar(xlabels, col_df_group['mean'], yerr=err_arr, color='tab:blue', capsize=3, ls='', marker='.')

    # annotate graph
    ax1.set_ylabel('count', color='dimgray')
    ax1.tick_params(axis='y', labelcolor='dimgray')
    ax1.set_xlabel(X.name)
    if not vars_new_df['numerical'].values[0] and xlabels.nunique()>6:
        ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_title(title_str)
    ax2.set_ylabel('success rate [-]', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    return vars_df, ax1, ax2

# define custom transformer that will reflect data about a specified mirror point
class SymmetricReflector(BaseEstimator, TransformerMixin):

    def __init__(self, x_mirror):
        # save inputs to instance
        self.x_mirror = x_mirror
        return

    def fit(self, X, y=None):
        # fit method is required structurally but no fitting is actually required
        # this class could be extended to calculate x_mirror to maximise correlation to target but for now will be elft to be defined manually
        return self
    
    def transform(self, X, y=None):
        # hack to turn data frame into series
        X = X.iloc[:,0]
        # mirror values
        X = (X-self.x_mirror[X.name]).abs()
        return X.values.reshape(-1, 1)

# define imputing and vectorising pipeline
def vect_pipe(cat_vars, num_vars, ord_vars, num_sym_vars, ord_mode_vars):

    # define individual transformers for each columns
    cat_trans = OneHotEncoder(handle_unknown='ignore', drop='if_binary')
    num_trans = StandardScaler()
    ord_trans = OrdinalEncoder(categories=list(ord_vars.values()))
    num_sym_trans = Pipeline([
        ('pre_vect', SymmetricReflector(x_mirror=num_sym_vars)),
        ('vect', StandardScaler())
        ])
    ord_mode_trans = Pipeline([
        ('pre_vect', SimpleImputer(missing_values='unknown', strategy='most_frequent')),
        ('vect', OrdinalEncoder(categories=list(ord_mode_vars.values())))
        ])

    # define column transformer
    col_trans = ColumnTransformer(
        transformers = [
            ('cat', cat_trans, cat_vars),
            ('num', num_trans, num_vars),
            ('ord', ord_trans, list(ord_vars.keys())),
            ('num_sym', num_sym_trans, list(num_sym_vars.keys())),
            ('ord_mode', ord_mode_trans, list(ord_mode_vars.keys())),
        ]
    )

    return col_trans

# extract column (e.g. `col`) from feature name (e.g. `col_value`)
def feat_2_col(feat, col_pot):
    # identify all potential columns
    col = [x for x in col_pot if feat.startswith(x)]
    # sort by longest (most specific) first
    col.sort(key=len, reverse=True)
    return col[0]

# extract feature names from column transformer
def get_feat_names(trans):

    # initialise blank lists
    feat_names = []
    col_names = []

    # loop over all sub transformers in the column transformer
    for sub_trans, cols in zip(trans.named_transformers_.values(), trans._columns): 
        
        # manually created pipeline has no method for feature extractions so extract inner vectoriser
        if sub_trans.__class__.__name__ in ['Pipeline']:
            sub_trans_use = sub_trans['vect']
        else:
            sub_trans_use = sub_trans

        # check class is one that that can be handled by this function
        class_name = sub_trans_use.__class__.__name__
        assert class_name in ['OneHotEncoder','StandardScaler','OrdinalEncoder']

        # extract feature and column names
        if class_name=='OneHotEncoder':
            feat_names_temp = sub_trans_use.get_feature_names_out(cols)
            feat_names.extend(feat_names_temp)
            col_names_temp = [feat_2_col(feat, cols) for feat in feat_names_temp]
            col_names.extend(col_names_temp)
        elif class_name in ['StandardScaler','OrdinalEncoder']:
            feat_names.extend(cols)
            col_names.extend(cols)

    return feat_names, col_names

# plot the accuracy and standard deviation for given subgroup
def grouped_df(ax, df, group):

    # calculate mean test score
    df = df[['mean_test_score','mean_train_score']+[group]].rename(columns={'mean_test_score':'Test', 'mean_train_score':'Train'})
    df = df.groupby(group, dropna=False).max()
    if df.index.dtype==float:
        df.index = df.index.map(lambda x: f'{x:.2f}')
    else:
        df.index = df.index.map(lambda x: str(x))

    # create bar plot
    sns.barplot(
        data=df.unstack().reset_index(),
        x=group,
        y=0,
        hue='level_0',
        ax=ax,
    )

    # decorate axis
    ax.legend(loc='center left', title=None)
    ax.set_ylabel('Mean Squared error')
    ax.set_title(group[6:])

    # annotate bar plot
    for rect_sub in ax.containers:
        for rect in rect_sub:
            height = rect.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(rect.get_x()+rect.get_width()/2, height),
                xytext=(0, -15),
                textcoords="offset points",
                ha='center',
                va='bottom'
            )

# drop unwanted columns from cv search results
def drop_columns(df):
    # identify columns to drop
    drop_cols = [col for col in df.columns if ((col.endswith('_time') and col!='mean_fit_time') or col.startswith('split') or col.startswith('rank_'))]
    # drop selected columns
    df.drop(columns=drop_cols, inplace=True)
    return df

# plot results for grouped dataframe
def grouped_plot(df):

    # initialise figure
    param_cols = [col for col in df.columns if col.startswith('param_')]
    no_params = len(param_cols)
    no_cols = min(no_params, 4)
    no_rows = int(np.ceil(no_params/no_cols))
    fig, ax_all = plt.subplots(ncols=no_cols, nrows=no_rows, figsize=(18,no_rows*3), squeeze=False)

    # loop through all groups
    for group, ax in zip(param_cols, ax_all.flatten()):
        grouped_df(ax, df, group)

    # define figure layout
    fig.tight_layout()

# calculate evaluation score
def eval_score(act, pred):

    # initialise score values
    att_value, pot_value = 0.0, 0.0

    # loop through each instance
    for act_i, pred_i in zip(act, pred):

        # update potential value
        if act_i:
            pot_value += 74

        # update attained value
        if pred_i:
            if act_i:
                att_value += 74
            else:
                att_value += -6

    return att_value/pot_value

# define custom metric
def revenue_attained(y_true, y_pred):

    # prepare data
    y_pred_bool = tf.convert_to_tensor(y_pred>0.5)
    y_true_bool = tf.cast(y_true, y_pred_bool.dtype)

    # calculate confusion matrix
    TP = tf.math.reduce_sum(tf.cast(tf.math.logical_and(y_pred_bool, y_true_bool),tf.int16))
    FN = tf.math.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred_bool), y_true_bool),tf.int16))
    FP = tf.math.reduce_sum(tf.cast(tf.math.logical_and(y_pred_bool, tf.math.logical_not(y_true_bool)),tf.int16))
    TN = tf.math.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred_bool), tf.math.logical_not(y_true_bool)),tf.int16))

    # calculate attained revenue
    attained_revenue = -6*FP + 74*TP

    # calculate potential revenue
    potential_revenue = 74*FN + 74*TP

    # calculate final score
    metric_value = attained_revenue/potential_revenue

    return metric_value

# define custom loss
class RevenueAttained(tf.keras.losses.Loss):
    def __init__(self, reduction=None, name=None):
        super().__init__()

    def call(self, y_true, y_pred):

        # prepare data
        y_true = tf.cast(y_true, y_pred.dtype)

        # calculate bounds 
        low = y_true*74
        high = 6*(1-y_true)

        # calculate loss
        lost_revenue = low*(1-y_pred) + high*y_pred
        loss = tf.math.reduce_mean(lost_revenue)

        return loss

# unit test for correlation functions
class test_corr_fun(unittest.TestCase):

    # test for numerical features
    def test_numerical(self):

        # calculate values
        X = pd.Series([0, 1, 2, 3, 4, 5, 6], name='test')
        y = pd.Series([0, 0, 0, 1, 1, 1, 1])
        df, title = corr_fun(X,y)

        # assert on values
        self.assertTrue(df['numerical'].values[0])
        self.assertTrue(abs(df['correlation_coefficient'].values[0]-0.866025)<0.001)
        self.assertTrue(abs(df['p-value'].values[0]-0.011725)<0.001)
        self.assertEqual(df.index[0], 'test')
        self.assertEqual(title, 'pointbiserialr correlation=0.866\npointbiserialr p-value=0.012')

    # test for categorical features
    def test_categorical(self):

        # calculate values
        X = pd.Series([0, 1, 2, 3, 4, 5, 6], name='test', dtype='category')
        y = pd.Series([0, 0, 0, 1, 1, 1, 1])
        df, title = corr_fun(X,y)

        # assert on values
        self.assertFalse(df['numerical'].values[0])
        self.assertTrue(pd.isnull(df['correlation_coefficient']).values[0])
        self.assertTrue(abs(df['p-value'].values[0]-0.320847)<0.001)
        self.assertEqual(df.index[0], 'test')
        self.assertEqual(title, 'Χ squared p-value=0.321')

# define genetic algorithm evolution
def ga_evolution(tb, pop, cx, mut):

    # select and clone the next generation individuals
    offspring = tb.select(pop, len(pop))
    offspring = list(map(tb.clone, offspring))

    # perform crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx:
            tb.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # mutate individuals
    for mutant in offspring:
        if random.random() < mut:
            tb.mutate(mutant)
            del mutant.fitness.values

    # evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(tb.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    fits = [ind.fitness.values[0] for ind in pop]

    # replace population with offspring
    pop[:] = offspring

    return pop, fits