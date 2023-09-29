import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import matplotlib
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


###################################################
# Business Problem:An educational platform derives course scores by using the ratings assigned to courses and employs these scores for ranking purposes.
# However, an employee argues that those who watch courses less tend to assign lower ratings, negatively impacting a fair scoring.
# Therefore, there is a desire to ascertain whether the extent of course viewing truly influences the scoring and to take action accordingly based on this determination.
###################################################

# Establishing Hypothesis
# H0: M1 = M2 (... there is no meaningful difference between two groups means.)
# H1: M1 != M2 (...there is ..)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()


############################
# Step-1: Normality Check
############################

# H0: Normal distrubuted.
# H1:..not normal distrubuted.

# if p-value < 0.05 then HO is rejected.
# if p-value < 0.05 then H0 cannot be rejected.

test_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p = 0.00 < 0.05 H0 is rejected, this group doesn't has normal distrubition

test_stat, pvalue = shapiro(df.loc[df["Progress"] < 25, "Rating"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p = 0.00 < 0.05 H0 is rejected,this group doesn't has normal distrubition


############################
# Step-2: Variance Homogenity Check
############################

# H0: Variances are homogenous
# H1: Variances are not homogenous

test_stat, pvalue = levene(df.loc[df["Progress"] > 75, "Rating"].dropna(),
                           df.loc[df["Progress"] < 25, "Rating"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p = 0.00 < 0.05 H0 rejected, variances are not homogenous


############################
# 1.2 Varianceas are not homogenous and groups don't have normal distrubition so mannwhitneyu testi (non-parametric test) is applied
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["Progress"] > 75, "Rating"].dropna(),
                                 df.loc[df["Progress"] < 25, "Rating"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p = 0.000 < 0.05 H0 rejected. There is a meaningful difference between two groups means.

#Analysis of the Results:

# The lower scores given by those who haven't watched most of the course compared to those who have watched most of it indicate that ;
#Ratings were given without knowing the content of the course, leading to a negative perception of the course.
# This statistically proven result suggests that the ratings of those who watch the course less should be given less weight
# compared to those who watch it more. For this purpose, coefficients can be determined based on viewing rates to calculate the ratings.

#For example;
#•	For 0-24%, use 0.20,
#•	For 25-49%, use 0.23,
#•	For 50-74%, use 0.27,
#•	For 75-100%, use 0.30
# By using these coefficients, we incorporate the viewing rate into the scoring. (0.20 + 0.23 + 0.27 + 0.30 = 1)


