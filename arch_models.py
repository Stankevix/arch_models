
# coding: utf-8

# # Concepts

# GARCH stands for "Generalized AutoRegressive Conditional Heteroskedasticity", and it is a popular approach to model volatility. Its based on https://campus.datacamp.com/courses/garch-models-in-python/

# ## What is volatility ?
# 
# 
# In finance, volatility is a statistical measure of the dispersion of asset returns over time. It is often computed as the standard deviation or variance of price returns. We will use the term "volatility" to describe standard deviation or variance interchangeably. Volatility describes uncertainties surrounding the potential price movement of financial assets. 
# 
# It is an essential concept widely used in risk management, portfolio optimization, and more. And it has been one of the most active areas of research in empirical finance and time series analysis. In general, the higher the volatility, the riskier a financial asset.
# 

# ### How to compute volatility?
# 
# We can compute the volatility as the standard deviation of price returns following three easy steps. 
# 
# * Step 1 is to calculate the returns as percentage price changes.
# * Step 2 is to calculate the sample mean return of a chosen n-period.
# * Step 3 is to derive the sample standard deviation. 
# 
# Also recall standard deviation is the square root of variance.

# ### Volatility conversion
# 
# Assume we measure volatility as the standard deviation of returns, then monthly volatility can be obtained by multiplying daily volatility by the square root of 21, which is the average number or trading days in a month. 
# 
# Similarly, annual volatility can be obtained by multiplying daily volatility by the square root of 252, which is the average number or trading days in a year.

# ### The Challenge
# 
# A common assumption in time series modeling is that volatility remains constant over time. However, heteroskedasticity, literally means "different dispersion" in ancient Greek, is frequently observed in financial return data. Volatility tends to increase or decrease systematically over time.

# ## ARCH & GARCH
# 
# Before GARCH, first came the ARCH models. ARCH stands for "Auto-Regressive Conditional Heteroskedasticity", and was developed by American economist Robert F. Engle in 1982. Here "conditional heteroscedasticity" means the data has time-dependent varying characteristic and unpredictable. Due to his contribution, Engle won the Nobel prize in economics in 2003.
# 
# Based on ARCH, GARCH models were developed by Danish economist Tim Bollerslev in 1986. The "G" in GARCH stands for "Generalized". Fun fact: Bollerslev wrote about the GARCH models in his Ph.D thesis, under the supervision of Engle, who was the inventor of ARCH models.

# ### White Noise
# 
# A time series is white noise if the variables are independent and identically distributed with a mean of zero. A residual is the difference between the observed value of a variable at time t and its predicted value based on information available prior to time t. 
# 
# If the prediction model is working properly, successive residuals are uncorrelated with each other, that is, they constitute a white noise time series. 
# 
# In other words, the model has taken care of all the predictable components of a time series, left only the unpredictable white noise part.

# ### GARCH(1,1) parameter constraints
# 
# To make a GARCH(1,1) process realistic, there are two conditions. First, it requires all the parameters, omega, alpha, and beta, to be non-negative. This ensures the variance can't be negative. Second, alpha plus beta should be less than one, which ensures the model estimated variance is always "mean-reverting" to the long-run variance. The long-run variance equals to omega divided by one minus alpha minus beta.

# ### GARCH(1,1) parameter dynamics
# 
# The rule of thumb regarding model parameter is: the larger the alpha, the bigger the immediate impact of the shocks. Here the shocks are expressed as residuals, or prediction errors. If we keep the alpha fixed, the larger the beta, the longer the duration of the impact, that is, high or low volatility periods tend to persist.

# ## Dataset & Libs

# In[42]:


import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as tsa
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
import math
import datetime as dt
import statsmodels.stats.api as sms

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace import sarimax
from sklearn.metrics import mean_squared_error
from arch import arch_model
from arch.univariate import ARX
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


sns.set_style('darkgrid')
warnings.filterwarnings("ignore")


# In[13]:


def read_datasets():
    df = create_index_datatime(pd.read_csv('itausa.csv',';'))
    df = df.interpolate()                                                 
    return df

def create_index_datatime(df):
    
    format = '%Y-%m-%d'
    
    df['data'] = pd.to_datetime(df['data'],format=format).dt.date
    df['anoMes'] = pd.to_datetime(df['data'],format=format).dt.strftime('%Y-%m')
    
    df = df.set_index('data')
    
    return df


# ## Stats Func Test

# In[22]:


def adf_test(dataset, log_test = False):
    ds = dataset
    
    if log_test:
        ds = np.log(ds)
        ds.dropna(inplace=True)
    
    alpha = 0.05
    
    result = tsa.adfuller(ds)
    print('Augmented Dickey-Fuller Test')
    print('test statistic: %.10f' % result[0])
    print('p-value: %.10f' % result[1])
    print('critical values')
    
    for key, value in result[4].items():
        print('\t%s: %.10f' % (key, value))
        
    if result[1] < alpha:  #valor de alpha é 0.05 ou 5 %
        print("Rejeitamos a Hipotese Nula")
    else:
        print("Aceitamos a Hipotese Nula")


# In[23]:


def normal_distribution_test(residual):
    print("\n normal_distribution_test \n ")
    print('p value of Jarque-Bera test is: ', stats.jarque_bera(residual)[1])
    print('p value of Shapiro-Wilk test is: ', stats.shapiro(residual)[1])
    print('p value of Kolmogorov-Smirnov test is: ', stats.kstest(residual, 'norm')[1])
    return


# In[24]:


def heteroscedasticity_test(results):
    print("\n heteroscedasticity_test \n ")
    print('p value of Breusch–Pagan test is: ', sms.het_breuschpagan(results.resid, results.model.exog)[1])
    #print('p value of White test is: ', sms.het_white(results.resid, results.model.exog)[1])
    return


# In[25]:


def varianceInflationFactor_test(df, target,exog):

    features = "+".join(df[exog].columns)
    print(features)
    
    t_target = target.join(" ~")
    
    y, X = dmatrices(t_target + features, df, return_type='dataframe')
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = [target]+exog
    
    return vif


# In[19]:


def print_corr(df,titulo):

    mask = np.triu(np.ones_like(df, dtype=bool))

    f, ax = plt.subplots(1,figsize=(10, 8))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(df, cmap=cmap, vmax=.9, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    ax.set(Title=titulo)

    plt.show()


# ## Model Functions

# In[7]:


def get_model_metrics(actual, forecast):
    
    rmse = np.sqrt(mean_squared_error(actual,forecast))
                
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
                
    num = 0
    den = actual.sum()
    for i in range(len(actual)):
        num = num + abs(actual[i]-forecast[i])
    wmape = 1 - num.sum()/den
    
    return rmse, mape, wmape


#     
#     # 1 - quando usamos todos os dados para ajustar o modelo, a estimativa do modelo
#     #tem viés de lookback. Na realidade, não sabemos o futuro, os dados da serie temporal usados
#     # para o ajuste do modelo e a previsão não devem se sobrepor.
#     
#     #2 - Esta abordagem esta menos sujeita a overfitting (sobreajuste). Uma suposição de modelagem
#     # de series temporais é que os parametros do modelo são estaveis ao longo do tempo.
#     #Mas isso não é verdade em um ambiente de mercado turbulento.
#     
#     #3 - Pode adaptar melhor a previsão a possiveis mudanças. Ao incorporar continuamente
#     # novas observações ao ajuste e preivsão do modelo, estamos mais abertos as condições
#     # do mercado como noticias, ciclos economicos etc.
# 

# In[156]:


def expanding_rw_forecast(model, df, start_loc,end_loc,n):
    forecasts = {}

    for i in range(n):
        # Specify fixed rolling window size for model fitting
        v_result = model.fit(first_obs = start_loc, 
                                 last_obs = i + end_loc, update_freq = 30)
        
        # Conduct 1-period variance forecast and save the result
        temp_result = v_result.forecast(horizon = 1).variance
        print("\n\n values - >", temp_result.values[-1, :])
        fcast = temp_result.iloc[i + end_loc]
        forecasts[fcast.name] = fcast
    
    # Save all forecast to a dataframe    
    forecast_var = pd.DataFrame(forecasts).T
    
    # Plot the forecast variance
    f, axes = plt.subplots(1, figsize=(15,5))
    plt.plot(forecast_var, color = 'red')
    plt.show()
    f, axes = plt.subplots(1, figsize=(15,5))
    plt.plot(df, color = 'green')
    plt.show()
    
    return forecast_var


# In[157]:


def fixed_rw_forecast(model, df, start_loc,end_loc,n):
    # Neste metodo, os pontos de dados antigos são eliminados da amostra
    # simultaneamente para manter um tamanho de janela fixa
    
    forecasts = {}

    for i in range(n):
        # Specify fixed rolling window size for model fitting
        v_result = model.fit(first_obs = i + start_loc, 
                                 last_obs = i + end_loc, update_freq = 30)
        
        # Conduct 1-period variance forecast and save the result
        temp_result = v_result.forecast(horizon = 1).variance
        print("\n\n values - >", temp_result.values[-1, :])
        fcast = temp_result.iloc[i + end_loc]
        forecasts[fcast.name] = fcast
    
    # Save all forecast to a dataframe    
    forecast_var = pd.DataFrame(forecasts).T
    
    # Plot the forecast variance
    f, axes = plt.subplots(1, figsize=(15,5))
    plt.plot(forecast_var, color = 'red')
    plt.show()
    f, axes = plt.subplots(1, figsize=(15,5))
    plt.plot(df, color = 'green')
    plt.show()    
    return forecast_var


# In[10]:


def get_arx_model(name,y,exog, lags):
    x=exog
    
    mod = arch_model(y,mean=name, lags=lags,dist="StudentsT")
    res = mod.fit(disp="off")
    print(res.summary())
    
    return mod,res


# In[69]:


def get_arch_model(df, p , q, mean, type_model, dist):
    
    #split_date = dt.date(2021,5,1)
    model = arch_model(df, p = p, q = q, mean = mean,vol=type_model,dist = dist)
    
    model_fit = model.fit(update_freq = 5)#last_obs=split_date
    
    print ("\n\n Model \n",model_fit.summary()) 
    #f, axes = plt.subplots(1, figsize=(15,5))
    model_fit.plot()
    plt.show()
    
    return model_fit, model #[-1:]


# ## Read Data

# In[14]:


itausa = read_datasets()


# In[38]:


itausa_train = itausa[0:1000]
itausa_test = itausa[1000:]


# In[15]:


itausa.head()


# In[30]:


plt.hist(itausa['last'],bins = 10, facecolor = 'green',label='Dist Last')


# ### Correlation

# In[20]:


itausa_corr = itausa.corr()

print_corr(itausa_corr,"ITAUSA")


# ### Stationary Test

# In[29]:


adf = adf_test(itausa['last'].diff().dropna())


# ### Observe volatility clustering
# 
# Volatility clustering is frequently observed in financial market data, and it poses a challenge for time series modeling.

# In[32]:


itausa['return_data'] = itausa['last'].pct_change()
itausa['volatility'] = itausa['return_data'].std()

f, axes = plt.subplots(1, figsize=(15,5))
plt.plot(itausa['return_data'], color = 'tomato', label = 'Daily last')
plt.legend(loc='upper right')
plt.show()


# In[41]:


std_daily = itausa['return_data'].std()
print('Daily volatility: ', '{:.2f}%'.format(std_daily))


# ### Multicollinearity

# In[34]:


target = 'last'
exog = ['year', 'month', 'open', 'max', 'min', 'Covid','return_data', 'volatility']

vif = varianceInflationFactor_test(itausa,target, exog)
vif


# ## Arch Modeling

# In[182]:


p = 15
q = 15
mean = 'Zero' #'zero', 'AR', 'constant'
type_model = 'GARCH' #'GARCH','ARCH', 'EGARCH'
dist = 't' #'t','skewt','normal'


model_fit, model = get_arch_model(itausa_train['last'],p, q, mean,type_model,dist)


# The null hypothesis is the parameter value is zero. If the p-value is larger than a given confidence level, the null hypothesis cannot be rejected, meaning the parameter is not statistically significant, hence not necessary.
# 
# Besides p-values, t-statistics can also help decide the necessity of model parameters. 
# 
# The t-statistic is computed as the estimated parameter value subtracted by its expected mean, and divided by its standard error. The absolute value of the t-statistic is a distance measure, that tells you how many standard errors the estimated parameter is away from 0. As a rule of thumb, if the t-statistic is larger than 2, you can reject the null hypothesis.

# In[183]:


para_summary = pd.DataFrame({'parameter':model_fit.params,
                             'p-value': model_fit.pvalues,
                             't-value': model_fit.tvalues,
                             'Log-likelihood': model_fit.loglikelihood,
                             'AIC':model_fit.aic,
                             'BIC':model_fit.bic})
para_summary


# ### Residual

# In[184]:


itausa_resid = model_fit.resid
itausa_std = model_fit.conditional_volatility

itausa_std_resid = itausa_resid/itausa_std


# In[185]:


# Plot the histogram of the standardized residuals
f, axes = plt.subplots(1, figsize=(10,5))
plt.hist(itausa_std_resid, facecolor = 'orange', label = 'Standardized residuals')
plt.show()


# In[186]:


f, axes = plt.subplots(1, figsize=(10,5))
plt.plot(itausa_std_resid)
plt.title('Standardized Residuals')
plt.show()


# In[187]:


# Generate ACF plot of the standardized residuals
plot_acf(itausa_std_resid, alpha = 0.05)
plt.show()


# Another powerful tool to check autocorrelations in the data is the Ljung-Box test.
# 
# The null hypothesis of Ljung-Box test is: 
# 
# * The data is independently distributed. If the p-value is larger than the specified significance level, the null hypothesis cannot be rejected. 
# * In other words, there is no clear sign of autocorrelations and the model is valid.

# In[188]:


# Perform the Ljung-Box test
lb_test = acorr_ljungbox(itausa_std_resid , lags = 10)

# Print the p-values
print('P-values are: ', lb_test[1])


# In[189]:


residual = model_fit.resid
normal_distribution_test(residual)
#heteroscedasticity_test(model_fit)


# ### Rolling window forecast

# **Expanding window forecast**
# 
# There are mainly two ways to perform a rolling window forecast. One is "expanding window" approach, which starts with a set of sample data, and as time moves forward, continuously adds new data points to the sample. 
# 
# Suppose we have 200 observations of a time-series. 
# 
# First, we estimate the model with the first 100 observations to forecast the data point 101. Then we include observation 101 into the sample, and estimate the model again to forecast the data point 102. 
# 
# The process is repeated until we have forecast for all 100 out-of-sample data points.
# 
# 
# Rolling window forecast is widely used because of the following motivations: 
# 
# * First, when we use all the data to fit a model, the model estimation has lookback bias. In reality, we do not know the future, so the time series data used for model fitting and forecast should not overlap. 
# 
# * Second, the rolling window approach is less subject to overfitting. An implicit time series modeling assumption is model parameters are stable over time. But this barely holds true in turbulent market environment. Imagine when we try to fit a GARCH(1,1) with observations from economic crisis versus normal market conditions, we are likely to obtain very different omega, alpha, beta results. 
# 
# * Third, the rolling window approach can better adapt our forecast to changes. By continuously incorporating new observations to the model fitting and forecast, we are more responsive to the most recent economic conditions, such as news, changes in economic cycles, etc.
# 
# **Fixed rolling window forecast**
# 
# Another rolling window forecast method is call "fixed rolling window forecast". Similarly it starts with a set of sample data, and as time moves forward, new data points are added. What different is old data points are dropped from the sample simultaneously to maintain a fixed window size.

# In[191]:


start_loc = 0
end_loc = 900
window = 100

forecast_exprw = expanding_rw_forecast(model,itausa_train['last'],start_loc,end_loc,window)


# In[113]:


forecast_fixrw = fixed_rw_forecast(model,itausa['last'],start_loc,end_loc,window)


# In[ ]:


# In[] Goodness of fit measures

