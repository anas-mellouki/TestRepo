import numpy as np
from math import sqrt
from wallstreet import Stock
import matplotlib.pyplot as plt


def data_retrieving(code_tickers):
    #Seeking the data for the th given tickers from Google Finance
    stocks = [Stock(code) for code in code_tickers]
    n_days_back = 30
    data_frames = [s.historical(days_back=n_days_back , frequency='d') for s in stocks]
    market_data_frame = (data_frames[0][["Date"]]).copy()
    for code, df in zip(code_tickers, data_frames):
        market_data_frame[code] = df.Close
    data_frame = market_data_frame.set_index("Date")
    return data_frame

code_tickers = ["AAPL", "AMZN", "FB", "GOOG", "TSLA", "NFLX"]
market_data_frame = data_retrieving(code_tickers)

def statistical_estimation(market_data_frame):
    #Function to estimate empirically the mean and covariances of the vector Y
    hist_y = (market_data_frame.copy()).iloc[:-1, :]
    for i in range(len(hist_y.index)):
        date = market_data_frame.index[i]
        next_date = market_data_frame.index[i+1]
        hist_y.loc[date] = market_data_frame.loc[next_date]/market_data_frame.loc[date]
    return hist_y.mean(), hist_y.cov()

class Market:
    #Class grouping the market parameters
    def __init__(self, r, mu, omega, date):
        self.p0 = p0
        self.r = r
        self.mu = mu
        self.omega = omega


N = len(code_tickers)
p0 = market_data_frame.iloc[-1, :]
r = 0.0001
mu, omega = statistical_estimation(market_data_frame)
market = Market(r, mu, omega, p0)

class Portfolio:
    #Class modeling the portfolio
    def __init__(self, a0, a, name=""):
        self.a0 = a0
        self.a = a
        self.name = name
        
    def value(self, prices):
        s = 0
        for quantity, price in zip(self.composition, prices):
            s += quantity*price
        return s
    
    def mean(self, market):
        wa = np.dot(np.diag(market.p0), self.a)
        return self.a0*(1+market.r) + np.dot(wa.transpose(), mu)
    
    def variance(self, market):
        wa = np.dot(np.diag(market.p0), self.a)
        return np.dot(wa.transpose(), np.dot(market.omega, wa))

def optimal_portfolio(market, v, sigma):
    #Returning a Portfolio object with the optimal composition for the given market parameters and sigma 
    N = len(market.mu)
    mu_tilde = market.mu - (1+r)
    if sigma > 0:
        lambda_star = sqrt(np.dot(mu_tilde.transpose(),np.dot(np.linalg.inv(omega), mu_tilde)))/sigma
        wa = (1/lambda_star)*np.dot(np.linalg.inv(omega), mu_tilde)
        a = np.dot(np.linalg.inv(np.diag(market.p0)), wa)
        e = np.ones(N)
        a0 = v - np.dot(wa.transpose(), e)
    else:
        a = np.zeros(N)
        a0 = v
    return Portfolio(a0, a)

def efficient_frontier(market, v, sigma_range):
    #Computing the efficient means for a given range of sigma values
    efficients = []
    for sigma in sigma_range:
        portfolio = optimal_portfolio(market, v, sigma)
        efficients.append(portfolio.mean(market))
    return efficients


def plot_portfolios(v, market):
    #Plotting the portfolios in the variance-mean space
    I = np.eye(N)
    unique_stock_portfolios = []
    for i,code in enumerate(code_tickers):
      a = (v/market.p0[i])*I[i,:]
      portfolio = Portfolio(0, a, code)
      unique_stock_portfolios.append(portfolio)
      
    min_sigma = 0
    min_return = v*(1+market.r)
    mu_tilde = market.mu - (1+market.r)
    e = np.ones(len(market.mu))
    sigma_star = v*sqrt(np.dot(mu_tilde.transpose(),np.dot(np.linalg.inv(market.omega), mu_tilde)))/np.dot(mu_tilde.transpose(),np.dot(np.linalg.inv(market.omega), e))
    mean_star = optimal_portfolio(market, v, sigma_star).mean(market)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['b','c', 'm', 'y', 'k', 'darkorange']
    
    sigma_max = 0
    for i, portfolio in enumerate(unique_stock_portfolios):
        sigma = sqrt(portfolio.variance(market))
        if sigma > sigma_max:
            sigma_max = sigma
        mean = portfolio.mean(market)
        ax.scatter(sigma,mean,marker="P",color=colors[i],s=500, label=portfolio.name)
    sigma_max = max(sigma_max, sigma_star)
    sigma_range = np.linspace(0, 2*sigma_max, num=10)
    means = efficient_frontier(market, v, sigma_range)    
    ax.scatter(sigma_star,mean_star,marker='*',color='r',s=500, label='Portefeuille $P^*$ 100% actif risqué')
    ax.scatter(min_sigma,min_return,marker='*',color='g',s=500, label='Portefeuille $P^0$ 100% actif non risqué')
    ax.vlines(sigma_star, 0.995*min(means), 1.005*max(means), colors='r', linestyles='dashed')
    ax.plot(sigma_range, means, linestyle='-.', color='black', label='Frontière efficiente')
    ax.set_title("Positionnement des portefeuilles dans l'espace moyenne variance")
    ax.set_xlabel('$\sigma$')
    ax.set_ylabel('$E[V_1]$')
    ax.legend(labelspacing=0.8)
    plt.savefig("plot.png")
    
v = 10
plot_portfolios(v, market)