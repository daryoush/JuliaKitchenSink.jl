

#https://juliaobserver.com/packages/TableReader
using TableReader
using DataFrames
using TimeZones
using Dates

url = "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"
dataframe = readcsv(`unzip -p data/oxfordmanrealizedvolatilityindices.zip oxfordmanrealizedvolatilityindices.csv `)


dataframe = DataFrames.rename(dataframe, :UNNAMED_1 => :timestring)
dateformat = Dates.DateFormat("yyyy-mm-dd HH:MM:SSzzz")
#
Dates.DateTime("2019-12-13 00:00:00+00:00", dateformat)
# 2019-12-13T00:00:00
dataframe.timestring[1]
Dates.DateTime(dataframe.timestring[1], dateformat)

dataframe.timestamp =  Dates.DateTime.(dataframe.timestring, dateformat)
symbols = unique(dataframe, :Symbol).Symbol
colnames  = names(dataframe)

coldict = collect(eachcol(dataframe, true))    # build a dict of col name and values
### OLD WAY!  see https://github.com/JuliaLang/julia/issues/17886
### namesOfIndicators =  [name for (name, value ) in filter(x -> eltype(x.second) == Float64, coldict)]

# https://realized.oxford-man.ox.ac.uk/documentation/estimators
# Available Estimators
# Code	Description
# bv	Bipower Variation (5-min)
# bv_ss	Bipower Variation (5-min Sub-sampled)
# close_price	Closing (Last) Price
# close_time	Closing Time
# medrv	Median Realized Variance (5-min)
# nobs	Number of Observations
# open_price	Opening (First) Price
# open_time	Opening Time
# open_to_close	Open to Close Return
# rk_parzen	Realized Kernel Variance (Non-Flat Parzen)
# rk_th2	Realized Kernel Variance (Tukey-Hanning(2))
# rk_twoscale	Realized Kernel Variance (Two-Scale/Bartlett)
# rsv	Realized Semi-variance (5-min)
# rsv_ss	Realized Semi-variance (5-min Sub-sampled)
# rv10	Realized Variance (10-min)
# rv10_ss	Realized Variance (10-min Sub-sampled)
# rv5	Realized Variance (5-min)
# rv5_ss	Realized Variance (5-min Sub-sampled)


#indicators are Float64 types (one seem to have missing value, fot now don't include it)
#(:medrv, Union{Missing, Float64})
namesOfIndicators = [name for (name, value) in coldict if eltype(value) == Float64]

#Using shifting arrays to deal with the deltas, note the potential to use
#regular loop also (instead of specialized panda code)
#https://discourse.julialang.org/t/create-lead-and-lag-variable-in-dataframe/8658/1

# There are multiple ways to calculate the Hurst index, below is the generalized
# version that is based on calculating the moments of the differences.

# spx['sqrt']= np.sqrt(spx['SPX2.rk'])
# spx['log_sqrt'] = np.log(spx['sqrt'])
#
# def del_Raw(q, x):
#     return [np.mean(np.abs(spx['log_sqrt'] - spx['log_sqrt'].shift(lag)) ** q)
#             for lag in x]
# note data is variance, need to take sqrt to get sigma, then log, then absolute
# value of diff at
#different lags and raised to power (for the qth moment) then mean

#  m(q,Δ)∝ Δ ** ζq    This means that the log of m(q, Δ) and ζ of q
# have linear relationship..   For each q the ζ is the slope.
# Hurst index is how the ζ changees with different q
# note  q is the q-moment of the delayed time series.

log_σ(σ_squre::Float64) = log(√σ_squ)

logσt(σ_squres) = log_σ.(σ_squres)
dataframe.rv10
dataframe.rv10
Define a striucture for the fraction information of a series
    Constructor takes a series and builds the parameters


# For start just get rv10 indicator for .SPX as timeseries

Function for timeseries (ts, qvecs,range of Delta,  )
    for each q moment
        for each delta

    return h, (for each q moment, zetaq, list of (log deltas, log m))
