# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:29:35 2020

@author: admin
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from statsmodels.tsa.seasonal import seasonal_decompose
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.linear_model import LinearRegression

import panel as pn
pn.extension()

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

#from flask import Flask, render_template

#Connect the app
#app = Flask(__name__)


# Linear Regression

PremiumLR = pd.read_csv("PremiumLR.csv")
X = PremiumLR.drop(columns = ["Premium"])
y = PremiumLR.iloc[:,1]
y = pd.DataFrame(y)
CorrDF = PremiumLR.corr(method = "pearson")

CorrDF.reset_index(inplace = True)
CorrDF.rename(columns = {"index":""}, inplace = True)
CorrDF = CorrDF.round(2)
#CorrDF

clfLR = LinearRegression()
LinReg = clfLR.fit(X, y)

lst = [246277, 187217, 165200]
X_test = pd.DataFrame(lst)
y_predict = clfLR.predict(X_test)
#y_predict

a = figure(plot_height = 250, plot_width = 350)
a.circle(X["Advisors"], y["Premium"], size = 8)
a.line(X["Advisors"], clfLR.predict(X)[:,0])
#show(a)

# STL Decomposition

PremiumES = pd.read_csv("PremiumES.csv")
PremiumES["Period"]= pd.to_datetime(PremiumES["Period"]) 
PremiumEStemp = PremiumES.set_index("Period")
#PremiumEStemp

decomp = decompose(PremiumEStemp, period = 4)

Obs = decomp.observed
Obs.reset_index(inplace = True)
Trend = decomp.trend
Trend.reset_index(inplace = True)
Seasonal = decomp.seasonal
Seasonal.reset_index(inplace = True)
Resid = decomp.resid
Resid.reset_index(inplace = True)

p = figure(x_axis_type = "datetime", plot_height = 250, plot_width = 350)
p.line(Obs["Period"], Obs["Premium"])
q = figure(x_axis_type = "datetime", plot_height = 250, plot_width = 350)
q.line(Trend["Period"], Trend["Premium"])
r = figure(x_axis_type = "datetime", plot_height = 250, plot_width = 350)
r.line(Seasonal["Period"], Seasonal["Premium"])
s = figure(x_axis_type = "datetime", plot_height = 250, plot_width = 350)
s.line(Resid["Period"], Resid["Premium"])

#show(column(p,q,r,s))

# Exponential Smoothing
HWmodel = ExponentialSmoothing(PremiumEStemp, trend = "add", seasonal="add", damped = False, seasonal_periods = 4).fit()
HWFit = pd.DataFrame(HWmodel.fittedvalues)
HWFit.reset_index(inplace = True)
HWFit.rename(columns = {0:"Premium"}, inplace = True)

b = figure(x_axis_type = "datetime", plot_height = 250, plot_width = 350)
b.line(PremiumES["Period"], PremiumES["Premium"])
b.line(HWFit["Period"], HWFit["Premium"])
#show(b)

# Final Forecasts

HWpred = HWmodel.forecast(3)
HWpred = pd.DataFrame(HWpred)
HWpred.reset_index(inplace = True)
HWpred = HWpred.rename(columns = {"index":"Period",0:"Stable Scenario"})
HWpred["Current Scenario"] = y_predict
HWpred = HWpred.round(2)
HWpred["Period"] = HWpred["Period"].dt.strftime("%d-%m-%Y")
#HWpred

# Dashboard presentation

DataCol = [TableColumn(field=Ci, title=Ci) for Ci in HWpred.columns] # bokeh columns
data_table = DataTable(columns=DataCol, source=ColumnDataSource(HWpred), height = 250, width = 350) # bokeh table

DataCol1 = [TableColumn(field=Ci1, title=Ci1) for Ci1 in CorrDF.columns] # bokeh columns
data_table1 = DataTable(columns=DataCol1, source=ColumnDataSource(CorrDF), height = 250, width = 350) # bokeh table

#gspec = pn.GridSpec(sizing_mode='stretch_both')

#gspec[0,0] = p
#gspec[1,0] = q
#gspec[2,0] = r
#gspec[3,0] = s
#gspec[0,1] = b
#gspec[1,1] = a
#gspec[2,1] = data_table1
#gspec[3,1] = data_table

#gspec.show()

pn.Column("Hosting my first app !!").servable()

#if __name__ == '__main__':
#    app.run(debug=False)

