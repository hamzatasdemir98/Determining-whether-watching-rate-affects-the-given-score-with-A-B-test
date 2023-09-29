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
# İş Problemi:Bir eğitim platformu, kurslara verilen puanları kullanarak kurs skorları elde ediyor ve bu puanları sıralamada kullanıyor.
# Ancak bir çalışan, kursları az izleyenlerin daha az puan verdiğini ve sağlıklı bir skorlamayı negatif etkilediğini iddia ediyor.
# Bu nedenle kursu izleme miktarının puanlamaya gerçekten bir etkisi olup olmadığı anlaşılarak buna göre aksiyon alınmak isteniyor.
###################################################

# Hipotezlerin Kurulması
# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()


############################
# A-Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p = 0.00 < 0.05 H0 reddedilir, normal dağılım yoktur

test_stat, pvalue = shapiro(df.loc[df["Progress"] < 25, "Rating"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p = 0.00 < 0.05 H0 reddedilir, normal dağılım yoktur


############################
# B-Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["Progress"] > 75, "Rating"].dropna(),
                           df.loc[df["Progress"] < 25, "Rating"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p = 0.00 < 0.05 H0 reddedilir, varyanslar homojen değildir


############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["Progress"] > 75, "Rating"].dropna(),
                                 df.loc[df["Progress"] < 25, "Rating"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p = 0.000 < 0.05 H0 reddedilir. Kursun çoğunu izleyenler ile izlemeyenlerin puanlamaları arasında anlamlı bir farklılık vardır.
