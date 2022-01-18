# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 19:01:14 2021

@author: Arnaud Ott 
"""




import numpy as np
import datetime as dt
from datetime import datetime
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.offline import plot





def call(strike, premium, buy_sell = "buy"):
    return lambda final_price : (max(final_price - strike, 0) - premium) * (1 if buy_sell == "buy" else -1)




def put(strike, premium, buy_sell = "buy"):
    return lambda final_price : (max(strike - final_price, 0) - premium) * (1 if buy_sell == "buy" else -1)




def compute_d1(r, S, K, T, sigma):
    return ( np.log(S/K) + (r + sigma**2/2) * T ) / ( sigma * np.sqrt(T) )
    




def get_delta(r, K, T, sigma, option = "call"):
    
    def delta(final_price):
        d1 = compute_d1(r, final_price, K, T, sigma)
        return  norm.cdf(d1, 0, 1)  if  option == "call" else norm.cdf(d1, 0, 1) - 1
        
    return delta








def get_gamma(r, K, T, sigma):
    
    def gamma(final_price):
        d1 = compute_d1(r, final_price, K, T, sigma)
        return norm.pdf(d1, 0, 1) / (final_price * sigma * np.sqrt(T))
    
    return gamma
  
    


    
  

def get_vega(r, K, T, sigma, stress = 0.01):
    
    def vega(final_price):        
        d1 = compute_d1(r, final_price, K, T, sigma)
        return final_price * norm.pdf(d1, 0, 1) * np.sqrt(T) * stress
    
    return vega 









def get_theta(r, K, T, sigma, option = "call"):
    
    def theta(final_price):
        d1    = compute_d1(r, final_price, K, T, sigma)
        d2    = d1 - sigma * np.sqrt(T)
        theta = - final_price * norm.pdf(d1, 0, 1) * sigma / ( 2 * np.sqrt(T) ) 
    
        if option == "call":
            theta += -r * K * np.exp(-r * T) * norm.cdf(d2, 0, 1) 
            
        else :
            theta += r * K * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
            
        return theta / 365
    
    return theta







def get_rho(r, K, T, sigma, option = "call", stress = 0.01):
    
    def rho(final_price):
        d1 = compute_d1(r, final_price, K, T, sigma)
        d2 = d1 - sigma*np.sqrt(T)
        
        if option == "call":
            rho = K * T * np.exp(-r * T) * norm.cdf(d2, 0, 1)
            
        else:
            rho = - K * T * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
            
        return rho * stress
    
    return rho







def find_nearest_value(array, value):
    array = np.asarray(array)
    index = ( np.abs(array - value) ).argmin()
    return array[index], index








    
def get_historical_volatility(historical_data):
    opendays   = len(historical_data)
    log_return = np.log( historical_data['Close'] / historical_data['Close'].shift(1) ).dropna()
    volatility = np.std( log_return ) * np.sqrt(opendays)
    return volatility    








def long_call(expiration_date, strike, asset, historical_volatility, volatility_type = "historical"):
    strike_list    = asset.option_chain(expiration_date)[0]["strike"] 
    strike, index  = find_nearest_value(strike_list, strike )
    premium        = asset.option_chain(expiration_date)[0]["lastPrice"][index]
    today          = dt.date.today()
    expiration_day = datetime.date( datetime.strptime(expiration_date, '%Y-%m-%d') )
    T              = (expiration_day - today).days / 365
    
    if (volatility_type == 'historical'):
        volatility = historical_volatility
    
    else: 
        volatility = asset.option_chain(expiration_date)[0]["impliedVolatility"][index]
    
    rho   = get_rho(0.01, strike, T, volatility, option='call', stress=0.01)    
    delta = get_delta(0.01, strike, T, volatility, option='call' )
    gamma = get_gamma(0.01, strike, T, volatility)
    vega  = get_vega(0.01, strike, T, volatility)
    theta = get_theta(0.01, strike, T, volatility)
    C     = call(strike, premium, buy_sell = "buy")
    
    return C, rho, delta, gamma, vega, theta
        






        
def short_call(expiration_date, strike, asset, historical_volatility, volatility_type = "historical"):
    strike_list    = asset.option_chain(expiration_date)[0]["strike"] 
    strike, index  = find_nearest_value(strike_list, strike )
    premium        = asset.option_chain(expiration_date)[0]["lastPrice"][index]
    today          = dt.date.today()
    expiration_day = datetime.date( datetime.strptime(expiration_date, '%Y-%m-%d') )
    T              = (expiration_day - today).days / 365
    
    if (volatility_type == 'historical'):
        volatility = historical_volatility
    
    else: 
        volatility = asset.option_chain(expiration_date)[0]["impliedVolatility"][index]
    
    rho   = get_rho(0.01, strike, T, volatility, option='call', stress=0.01)    
    delta = get_delta(0.01, strike, T, volatility, option='call' )
    gamma = get_gamma(0.01, strike, T, volatility)
    vega  = get_vega(0.01, strike, T, volatility)
    theta = get_theta(0.01, strike, T, volatility)
    C     = call(strike, premium, buy_sell = "sell")
    
    return C, rho, delta, gamma, vega, theta        
    








def long_put(expiration_date, strike, asset, historical_volatility, volatility_type = "historical"):
    strike_list    = asset.option_chain(expiration_date)[1]["strike"] 
    strike, index  = find_nearest_value(strike_list, strike )
    premium        = asset.option_chain(expiration_date)[1]["lastPrice"][index]
    today          = dt.date.today()
    expiration_day = datetime.date( datetime.strptime(expiration_date, '%Y-%m-%d') )
    T              = (expiration_day - today).days / 365
    
    if (volatility_type == 'historical'):
        volatility = historical_volatility
    
    else: 
        volatility = asset.option_chain(expiration_date)[1]["impliedVolatility"][index]
    
    rho   = get_rho(0.01, strike, T, volatility, option='put', stress=0.01)    
    delta = get_delta(0.01, strike, T, volatility, option='put' )
    gamma = get_gamma(0.01, strike, T, volatility)
    vega  = get_vega(0.01, strike, T, volatility)
    theta = get_theta(0.01, strike, T, volatility)
    P     = put(strike, premium, buy_sell = "buy")
    
    return P, rho, delta, gamma, vega, theta

    






def short_put(expiration_date, strike, asset, historical_volatility, volatility_type = "historical"):
    strike_list    = asset.option_chain(expiration_date)[1]["strike"] 
    strike, index  = find_nearest_value(strike_list, strike )
    premium        = asset.option_chain(expiration_date)[1]["lastPrice"][index]
    today          = dt.date.today()
    expiration_day = datetime.date( datetime.strptime(expiration_date, '%Y-%m-%d') )
    T = (expiration_day - today).days / 365
    
    if (volatility_type == 'historical'):
        volatility = historical_volatility
    
    else: 
        volatility = asset.option_chain(expiration_date)[1]["impliedVolatility"][index]
    
    rho   = get_rho(0.01, strike, T, volatility, option='put', stress=0.01)    
    delta = get_delta(0.01, strike, T, volatility, option='put' )
    gamma = get_gamma(0.01, strike, T, volatility)
    vega  = get_vega(0.01, strike, T, volatility)
    theta = get_theta(0.01, strike, T, volatility)
    P     = put(strike, premium, buy_sell = "sell")
    
    return P, rho, delta, gamma, vega, theta









def compute_break_even(payoff, spot_price, final_price_list):
    break_even_list = list()
    reference_value = payoff[0] 
    
    for index in range(1, len(payoff)):
        if(reference_value * payoff[index] ) <0:
            break_even_list.append(index)
            reference_value = payoff[index]

    break_even_deviation = np.around( (final_price_list[break_even_list[:] ] / spot_price - 1 ) * 100, 3 )
    break_even_percentage = list()
    
    for break_even in break_even_deviation:
        break_even_percentage.append(str(break_even  ) + "%" )
    
    return break_even_list, break_even_percentage     
   









def strategy_creation(expiration_date = '2024-01-19', long_call_list = [], short_call_list = [], \
                      long_put_list = [], short_put_list = [],\
                      asset_name = "AAPL", volatility_type = "historical"):
    

    long_call_number  = len(long_call_list)
    short_call_number = len(short_call_list)
    long_put_number   = len(long_put_list)
    short_put_number  = len(short_put_list)
    
    
    asset                 = yf.Ticker(asset_name)
    historical_data       = asset.history('1y')
    historical_volatility = get_historical_volatility(historical_data)
    spot_price            = historical_data['Close'][-1]
    strike_list           = asset.option_chain(expiration_date)[1]["strike"]
    

    final_price_list = np.linspace(min(strike_list), max(strike_list), 500)
    
    
    
    strategy_payoff = list(0 for i in range(0, len(final_price_list) ) )
    strategy_delta  = list(0 for i in range(0, len(final_price_list) ) )
    strategy_gamma  = list(0 for i in range(0, len(final_price_list) ) )
    strategy_theta  = list(0 for i in range(0, len(final_price_list) ) )
    strategy_rho    = list(0 for i in range(0, len(final_price_list) ) )
    strategy_vega   = list(0 for i in range(0, len(final_price_list) ) )

    
    if (long_call_number != 0):
        for i in range(long_call_number):
            strike = long_call_list[i]
            C, rho, delta, gamma, vega, theta = long_call(expiration_date, strike, asset, historical_volatility, volatility_type )
            
            for index in range(len(final_price_list)):
                strategy_payoff[index] += C(final_price_list[index])
                strategy_delta[index]  += delta(final_price_list[index])
                strategy_gamma[index]  += gamma(final_price_list[index])
                strategy_theta[index]  += theta(final_price_list[index])
                strategy_rho[index]    += rho(final_price_list[index])
                strategy_vega[index]   += vega(final_price_list[index])
                 
                                
    if (short_call_number != 0):
        for i in range(short_call_number):
            strike = short_call_list[i]
            C, rho, delta, gamma, vega, theta = short_call(expiration_date, strike, asset, historical_volatility, volatility_type )
            
            for  index in range(len(final_price_list)):
                strategy_payoff[index] += C(final_price_list[index])
                strategy_delta[index]  -= delta(final_price_list[index])
                strategy_gamma[index]  -= gamma(final_price_list[index])
                strategy_theta[index]  -= theta(final_price_list[index])
                strategy_rho[index]    -= rho(final_price_list[index])
                strategy_vega[index]   -= vega(final_price_list[index])

                                         
    if (long_put_number != 0):
        for i in range(long_put_number):
            strike = long_put_list[i]
            P, rho, delta, gamma, vega, theta = long_put(expiration_date, strike, asset, historical_volatility, volatility_type )
                  
            for  index in range(len(final_price_list)):
                strategy_payoff[index] += P(final_price_list[index])
                strategy_delta[index]  += delta(final_price_list[index])
                strategy_gamma[index]  += gamma(final_price_list[index])
                strategy_theta[index]  += theta(final_price_list[index])
                strategy_rho[index]    += rho(final_price_list[index])
                strategy_vega[index]   += vega(final_price_list[index])  
     
                                  
    if (short_put_number != 0):
        for i in range(short_put_number):
            strike = short_put_list[i]
            P, rho, delta, gamma, vega, theta = short_put(expiration_date, strike, asset, historical_volatility, volatility_type )
            
            for  index in range(len(final_price_list)):
                strategy_payoff[index] += P(final_price_list[index])
                strategy_delta[index]  -= delta(final_price_list[index])
                strategy_gamma[index]  -= gamma(final_price_list[index])
                strategy_theta[index]  -= theta(final_price_list[index])
                strategy_rho[index]    -= rho(final_price_list[index])
                strategy_vega[index]   -= vega(final_price_list[index])

     
    
    return final_price_list, strategy_payoff, strategy_delta, \
            strategy_gamma, strategy_theta, strategy_rho, strategy_vega,\
            asset_name, spot_price, expiration_date     













def interactive_strategy_plot(final_price_list, payoff, delta, gamma, theta, rho, vega, 
                              asset_name, spot_price, expiration_date ):
   
    title=' Option Strategy (' + asset_name +'), Exp: '+ expiration_date
    
    break_even_index, break_even_percentage = compute_break_even(payoff, spot_price, final_price_list)
    
    delta            = [100 * i for i in delta]
    gamma            = [100 * i for i in gamma]
    theta            = [100 * i for i in theta]
    rho              = [100 * i for i in rho]
    vega             = [100 * i for i in vega]
    
    payoff_array = np.array(payoff)
    size_list = np.ones((len(break_even_index)))*30
    
    fig = go.Figure(data = go.Scatter(
        
        mode          = 'markers+text',
        x             = final_price_list[ break_even_index ],
        y             = payoff_array[ break_even_index ],
        marker        = dict(size=size_list, color = "black"),
        name          = "Break even",
        text          = break_even_percentage,
        textposition  = "top right",
        textfont      = dict(size=30,color = "black"),
    
        )
     )
    
    
    profit_index = payoff_array >= 0
    
    fig.add_trace( go.Scatter(x    = final_price_list,
                              y    = payoff_array,
                              name = 'Payoff',
                              line = dict(width=10,color='blue')))
    
    
    fig.add_trace( go.Scatter(x    = final_price_list[profit_index],
                              y    = payoff_array[profit_index],
                              name = 'Profit',
                              fill = 'tozeroy', 
                              line = dict(color='green')))
    
    
    fig.add_trace( go.Scatter(x    = final_price_list[~profit_index],
                              y    = payoff_array[~profit_index],
                              name = 'Loss',
                              fill = 'tozeroy', 
                              line = dict(color='red')))
    
    
    fig.add_trace( go.Scatter(x    = final_price_list,
                              y    = delta,
                              name = 'Delta',
                              line = dict(color='royalblue'),
                              visible=False)
                  )
    
    fig.add_trace( go.Scatter(x    = final_price_list,
                              y    = gamma,
                              name = 'Gamma',
                              line = dict(color='fuchsia'),
                              visible=False)
                  )
    
    fig.add_trace( go.Scatter(x    = final_price_list,
                              y    = theta,
                              name = 'Theta',
                              line = dict(color='mediumspringgreen'),
                              visible=False)
                  )
    
    fig.add_trace( go.Scatter(x    = final_price_list,
                              y    = rho,
                              name = 'Rho',
                              line = dict(color='goldenrod'),
                              visible=False)
                  )
    
    fig.add_trace( go.Scatter(x    = final_price_list,
                              y    = vega,
                              name = 'Vega',
                             line  = dict(color='chocolate'),
                             visible=False)
                  )
    
    
    updatemenus = list([
        dict(active = 0,
             buttons = list([   
                dict(label ='Payoff',
                     method = 'update',
                     args = [{'visible': [True, True, True, True, False, False, False, False, False]},
                             {'title': title +' Profit and Loss',
                              'xaxis': {'title': 'Stock Price - Dollars'},
                              'yaxis': {'title': 'Profit and Loss - Dollars'}}
                             ]),
    
                dict(label = 'Delta',
                     method = 'update',
                     args = [{'visible': [False, False, False, False, True, False, False, False, False ]},
                             {'title': title +' Delta',
                              'xaxis': {'title': 'Stock Price - Dollars'},
                              'yaxis': {'title': 'Delta (%)'}}                       
                              ]),
                
                dict(label = 'Gamma',
                     method = 'update',
                     args = [{'visible': [False, False, False, False, False, True, False, False, False ]},
                             {'title': title +' Gamma',
                              'xaxis': {'title': 'Stock Price - Dollars'},
                              'yaxis': {'title': 'Gamma (%)'}}                       
                              ]),
                
                dict(label = 'Theta',
                     method = 'update',
                     args = [{'visible': [False, False, False, False, False, False, True, False, False ]},
                             {'title': title +' Theta',
                              'xaxis': {'title': 'Stock Price - Dollars'},
                              'yaxis': {'title': 'Theta (%)'}}                       
                              ]),     
                
                dict(label = 'Rho',
                     method = 'update',
                     args = [{'visible': [False, False, False, False, False, False, False, True, False ]},
                             {'title': title +' Rho',
                              'xaxis': {'title': 'Stock Price - Dollars'},
                              'yaxis': {'title': 'Rho (%)'}}                       
                              ]), 
                
                dict(label = 'Vega',
                     method = 'update',
                     args = [{'visible': [False, False, False, False, False, False, False, False, True ]},
                             {'title': title +' Vega',
                              'xaxis': {'title': 'Stock Price - Dollars'},
                              'yaxis': {'title': 'Vega (%)'}}                       
                              ]),             
                
            ]),
        )
    ])
    
    
    fig.update_layout(
        title={
        'text': title,
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend  = True,
        font        = dict(size=20, color="black"),
        updatemenus = updatemenus,
        template    = "plotly_white",
        font_family = "Palatino Linotype"
    )
    
    fig.add_vline(x=spot_price, annotation_text = "Spot Price: " + str(round(spot_price, 2)) + "$", line_width = 3, line_dash = "dash", line_color = "black")
    
    
    plot(fig, show_link = False)    












if __name__ == '__main__':
    
    asset_name = 'AAPL'
    asset = yf.Ticker(asset_name)
    
    print("The different expiration dates for the asset " + asset_name + " are :" )
    print( asset.options)
    print("The different strikes for the asset " + asset_name + " are :" )
    print( asset.option_chain()[0]["strike"])
    
    
    
    strategy = strategy_creation(expiration_date = '2024-01-19', long_call_list=[125,60], short_call_list=[100, 75], 
                      long_put_list=[150,220], short_put_list=[175, 200],\
                      asset_name = "AAPL", volatility_type = "historical")
        
    final_price_list = strategy[0]    
    payoff           = strategy[1]
    delta            = strategy[2]
    gamma            = strategy[3]
    theta            = strategy[4]
    rho              = strategy[5]
    vega             = strategy[6]
    asset_name       = strategy[7]
    spot_price       = strategy[8]
    expiration_date  = strategy[9]

    
    interactive_strategy_plot(final_price_list, payoff, delta, gamma, theta, rho, vega, 
                              asset_name, spot_price, expiration_date )

        
    
