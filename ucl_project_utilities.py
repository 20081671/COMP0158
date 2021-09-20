import warnings
import os
import time
import pandas as pd
import numpy as np
import mppca as mppca
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from importlib import reload
from scipy.stats import norm
reload(mppca)

path = __file__
ts = os.path.getmtime(path)
dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print(path,'@',dt)    

def load_and_preprocess(ccy='USD', files='old', skew_data=False):
    '''Load and preprocess the raw .csv data'''
    
    # Get old or new files
    if files=='old':
        file_atm = r'\\auksinfra031.file.core.windows.net\appdata\quant\swaption data\swaption_final_snap_straddle_'+ccy+'.csv' 
        file_skew = r'\\auksinfra031.file.core.windows.net\appdata\quant\swaption data\swaption_final_snap_'+ccy+'.csv' 
    elif files=='new':
        file_atm = r'\\auksinfra031.file.core.windows.net\appdata\quant\swaption data\latest_snap_csvs\swaption_final_snap_straddle_'+ccy+'.csv' 
        file_skew = r'\\auksinfra031.file.core.windows.net\appdata\quant\swaption data\latest_snap_csvs\swaption_final_snap_'+ccy+'.csv' 
    
    t0 = time.time()
    
    print('Loading atm data...')
    df_all = pd.read_csv(file_atm)
    
    if skew_data:
        print('Loading skew data...')
        df_skew = pd.read_csv(file_skew)
    
        print('Collating data...')
        df_all = df_all.append(df_skew)
    
    print('Loading time:', round(time.time()-t0,1), 'seconds')
    
    t0 = time.time()
    
    # Add numeric expiries/tenors
    exp_num = []
    for T in df_all['Expiry']:
        d = 12 if T[-1] == 'M' else 1
        exp_num.append(round(float(T[:2])/d,2))
    ten_num = []
    for T in df_all['Tenor']:
        d = 12 if T[-1] == 'M' else 1
        ten_num.append(round(float(T[:2])/d,2))

    expiries = np.unique(exp_num)
    tenors = np.unique(ten_num)
    strikes = np.unique(df_all['Relative Strike'])
        
    df_all.insert(7, column='Expiry Num', value=exp_num)
    df_all.insert(8, column='Tenor Num', value=ten_num)

    # Add index of Expiries x Tenors (ExT) for easier sorting later on
    n_expiries = len(expiries)
    n_tenors = len(tenors)
    n_strikes = len(strikes)
    idx = []
    for z in zip(df_all['Relative Strike'], exp_num, ten_num):
        idx.append(strikes.tolist().index(z[0]) * n_tenors * n_expiries + expiries.tolist().index(z[1]) * n_tenors + tenors.tolist().index(z[2]))

    df_all.insert(9, column='idx', value=idx)
    
    # Simplify dates
    df_all.insert(1, column='Date', value=df_all['Date Time (UTC)'].str[:10]) # Bc we have skipped the datetime parsing for speed
    df_all['Date'] = pd.to_datetime(df_all['Date']) # Far quicker than parsing inside read_csv()
    
    # Clean up rows/columns
    # TODO: look into the strike=0 issue in the 'skew' data
    # TODO: clean up the instruments as needed in all currencies
    df_all = df_all[~df_all['Instrument'].str.contains("SOSWOUSR")] # Remove the (very few) rows with SOFR instruments e.g. SOSWOUSR01M01Y
    df_all = df_all[~(df_all['Strike Price']==0)] # 5 of them have K=0, remove
    df_all.drop(['Unnamed: 0'], axis=1, inplace=True)   
    
    # Clean up "0" vols
    eps = 1
    df_all.loc[df_all['Normal Volatility IBOR discounted']<=eps,'Normal Volatility IBOR discounted'] = np.nan
    df_all.loc[df_all['Normal Volatility OIS discounted Calendar Days']<=eps,'Normal Volatility OIS discounted Calendar Days'] = np.nan
    df_all.loc[df_all['Lognormal Volatility IBOR discounted']<=eps,'Lognormal Volatility IBOR discounted'] = np.nan
    df_all.loc[df_all['Lognormal Volatility OIS discounted']<=eps,'Lognormal Volatility OIS discounted'] = np.nan    
    
    # TODO: outliers
    
    # Combine vol data    
    # TODO: this is only a temporary design decision, look into it
    # For ATM data:
    #df_all['Vol'] = df_all['Normal Volatility IBOR discounted'] 
    #df_all['Vol'] = df_all['Normal Volatility OIS discounted Calendar Days']
    #df_all['Vol'] = df_all['Lognormal Volatility OIS discounted'] 
    
    # Logn-vol seems to be the more reliable data
    # we'll use it as a base to build a synthetic normal vol dataset
    # TODO: need to review the data building process
    T = df_all['Expiry Num'].values
    K = df_all['Strike Price'].values
    f = K
    lv = df_all['Lognormal Volatility IBOR discounted'].values
    nvs = lognormal_to_normal(lv/100, T, K, f) * 100
    df_all['Vol'] = nvs
    # Augment with normal vol if available where the transform is missing
    df_all.loc[df_all.Vol.isna() & ~df_all['Normal Volatility IBOR discounted'].isna(),'Vol'] = df_all[df_all.Vol.isna() & ~df_all['Normal Volatility IBOR discounted'].isna()]['Normal Volatility IBOR discounted'].values
    if skew_data:
        # For Skew data:
        df_all.loc[df_all.Vol.isna(),'Vol'] = df_all.loc[df_all.Vol.isna(),'Skew Norm Vol']
        
    
    # Sort df and reset index
    df_all.sort_values(['Date', 'Expiry Num', 'Tenor Num', 'Relative Strike'], inplace=True)
    df_all.reset_index(level=0, inplace=True, drop=True)
    
    print('Prepping time:', round(time.time()-t0,1), 'seconds')
    
    return df_all, expiries, tenors, strikes
    
def remove_outliers(X0, window=5, num_sd=3, obs_pct=0.1):
    '''Remove series outliers if rolling window z-score is above
    num_sd AND no more than obs_pct of the observation are deemed 
    outliers'''
    X = X0.copy()
    N, D = X.shape
    idx_outlier = np.zeros((N,D)).astype(bool)
    for n in range(N):
        idx_from, idx_to = np.maximum(0,n-window), np.minimum(N,n+1+window)
        point = X[n,:]
        data_before = X[idx_from:n,:].reshape(-1,D)
        data_after = X[n+1:idx_to,:].reshape(-1,D)
        ref_data = np.vstack([data_before, data_after])
        m, s = np.mean(ref_data,axis=0), np.std(ref_data,axis=0)    
        idx_outlier[n] = (np.abs((point - m) / s) > num_sd)
    
    pct_outliers = idx_outlier.sum(axis=1) / D 
    
    for d in range(D):
        idx = (pct_outliers <= obs_pct) & idx_outlier[:,d]
        X[idx,d] = np.nan
    
    return X
 
def data_density(df, K, expiries, tenors, cutoff=0, style='absolute'):
    '''Compute the data density (fraction of the date range with valid data)
    for each expiry and tenor at a given strike'''
 
    N = len(df['Date'].unique())
    df1 = df[df['Relative Strike'] == K]    
    eps = 1e-9
    density = np.zeros((len(expiries), len(tenors)))
    for r,e in enumerate(expiries):
        for c,t in enumerate(tenors):
            df2 = df1[(df1['Expiry Num']==e) & (df1['Tenor Num']==t)]
            density[r,c] = np.sum(df2.Vol > cutoff) 
            if style == 'absolute':
                density[r,c] /= N+eps
            elif style=='relative':
                density[r,c] /= len(df2)+eps
            else:
                raise StyleError("Valid styles: 'absolute', 'relative'")
    
    return density
    
    
def display_pcs(pca, expiries, tenors, M=3, zlim=None, show_variance=True):
    '''Display first M principal components and their explained variance'''

    n_expiries = len(expiries)
    n_tenors = len(tenors)
    X, Y = np.meshgrid(expiries, tenors)
    
    n_cols = 4
    n_rows, dr = divmod(M+1, n_cols)
    n_rows += 1 if dr > 0 else 0
    fig = plt.figure(figsize = [20,5*n_rows])
        
    for m in range(M):
        ax = fig.add_subplot(n_rows, n_cols, m+1, projection='3d')
        PC_flat = pca.pcs_[m] * np.sign(pca.pcs_[m][0]) #pca.components_[m]
        PC = PC_flat.reshape(n_expiries, n_tenors)
        ax.plot_wireframe(X, Y, PC.T)
        if zlim is None:
            ax.set_zlim([np.nanmin(PC) - 0.01, np.nanmax(PC) + 0.01])
        else:
            ax.set_zlim(zlim)
        ax.set_xlabel('Expiry')
        ax.set_ylabel('Tenor')
        ax.set_title('$p_{}$'.format(m+1)+', (cumul. variance '+str(np.round(100*np.cumsum(pca.explained_variance_ratio_)[m],1))+'%)')
    
    if show_variance:
        ax = fig.add_subplot(n_rows, n_cols, m+2)
        ax.plot(np.cumsum(pca.explained_variance_ratio_[:M]),'o-')
        ax.set_title('Explained variance ratio')
        
    return None


def extract_pcs(W, s2):
    '''Extract principal components U from WW.T = U(L-vI)U.T
    (see Bishop, Tipping 99)'''
    
    U = np.zeros_like(W)
    E = np.zeros((W.shape[0],W.shape[2]))
    
    for p in range(W.shape[0]):
        #Eiv0, R_T = np.linalg.eigh(W[p].T @ W[p])
        #Eiv0 = Eiv0[::-1]
        #R_T = R_T[:,::-1]
        #U[p] = W[p] @ np.linalg.inv(np.diag(Eiv0)**0.5 @ R_T.T)
        #E[p] = Eiv0
        Q = W[p].shape[1]
        ev, eV = eigh(W[p] @ W[p].T)
        U[p] = eV[:,:Q]
        E[p] = ev[:Q] + s2[p]
    
    return U, E
    
    
def lognormal_to_normal(logn_vol, T, K, f):
    '''Convert lognormal vol to normal vol, with expiry T, strike K, forward f.
    Implementation based on Hagan's "Volatility Conversion Calculators"
    http://janroman.dhis.org/finance/Norm%20-%20LogNorm/Hagan%20Normvol.pdf'''

    logn_vol = np.array(logn_vol)
    T = np.array(T)
    K = np.array(K)
    f = np.array(f)
    
    n_vol = np.zeros(len(logn_vol))
    
    idx = np.abs(f/K-1) < 0.001
    n_vol_atm = logn_vol * np.sqrt(f*K) * (1 + 1/24 * np.log(f/K)**2) / (1 + 1/24 * logn_vol**2 * T + 1/5760 * logn_vol**4 * T**2)
    
    # This part will return 0/0 = nan where f==K, so we only do the f!=K cases
    f = f[~idx]
    K = K[~idx]
    T = T[~idx]
    logn_vol = logn_vol[~idx]
    if len(f) > 0:
        denom = 1 + 1/24 * (1 - 1/120 * np.log(f/K)**2) * logn_vol**2 * T + 1/5760 * logn_vol**4 * T**2
        n_vol_skew = logn_vol * (f-K) / np.log(f/K) * 1 / denom
        n_vol[~idx] = n_vol_skew
    # TODO: just re-review this for f<>K, does the scale of (f-K) matter vs atm case ? 
    # Think not as sqrt(f*K*100*100) = scaling by 100, same as 100*(f-K) vs (f-K), 
    # so using one or the other formula will return results with the same OOM
    
    n_vol[idx] = n_vol_atm[idx]   
        
    return n_vol_atm * idx + n_vol * (1-idx)
    
def get_test_idx(pop_size, n_groups=1, group_size=1, seed=None, force_n_last=0):
    rng = np.random.default_rng(seed)
    N = (pop_size-force_n_last) // group_size
    idx_g = np.sort(rng.choice(N, n_groups, replace=False))
    
    idx_n = []    
    for i in idx_g:
        idx_n += list(range(i*group_size, (1+i)*group_size))
    
    if force_n_last > 0:
        idx_last = list(range(pop_size-force_n_last, pop_size))
        idx_n += idx_last
    
    return np.array(idx_n) 
    
def get_residuals(X, U, q=3):
    '''Compute residuals as X - X @ U @ U.T
    U: eigenvectors from PCA in columns, leftmost is PC1.'''    
    X_rec = X @ U[:,:q] @ U[:,:q].T
    return X - X_rec
 
def Sharpe(cum_pnl, time_scale=np.sqrt(252), ci_level=0.99, return_ci=False, round_sr=4):
    '''Sharpe ratio and confidence interval around it'''
    pnl = np.diff(cum_pnl,axis=0)
    sr = pnl.mean(axis=0) / pnl.std(axis=0) * time_scale
    # Confidence interval based on (Lo 2002)
    N = cum_pnl.shape[0]
    sr_lo = sr + np.sqrt((1+sr**2)/(N-1)) * norm.ppf((1-ci_level)/2)
    sr_hi = sr + np.sqrt((1+sr**2)/(N-1)) * norm.ppf(1-(1-ci_level)/2)
    if return_ci:
        return np.round(sr,round_sr), np.round(sr_lo,round_sr), np.round(sr_hi,round_sr)
    else:
        return np.round(sr,round_sr)
        
def interp_nan(x):
    '''Interpolate nan in 1D series using closest valid values'''
    x_int = x.copy()
    idx_valid = np.nonzero(~np.isnan(x))[0]
    for i in range(len(x)):
        if np.isnan(x[i]):
            if (np.min(idx_valid)<i) and (np.max(idx_valid)>i):
                idx_a = np.max(idx_valid[idx_valid < i])
                idx_b = np.min(idx_valid[idx_valid > i])
                x_int[i] = x[idx_a] + (x[idx_b] - x[idx_a]) * (i-idx_a)/(idx_b-idx_a)
    return x_int

def ext_to_idx(e,t,expiries,tenors):
    idx_e = expiries.index(e)
    idx_t = tenors.index(t)
    return idx_e * len(tenors) + idx_t
    
def idx_to_ext(idx,expiries,tenors):
    idx_e, idx_t = divmod(idx,len(tenors))
    return expiries[idx_e], tenors[idx_t]

def info():
    path = __file__
    ts = os.path.getmtime(path)
    dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print('module timestamp:',dt)    

def eigh(C):
    '''A version of np.linalg.eigh that returns the 
    eigenvalues in decreasing not increasing order'''
    ev, eV = np.linalg.eigh(C); ev = ev[::-1]; eV = eV[:,::-1]
    return ev, eV
    
def make_boxplot(X,expiries,ylim=None,title='',xlabel='',hline=None, sym='b+',axes=None):
    '''A useful way to look at data'''
    exp_label = []
    exp_idx = []
    eps_dt = 5/365 # Because int() creates problems
    for e in expiries:
        if int(e) != e:
            exp_label.append('{}m'.format(int((e+eps_dt)*12)))
        else:
            exp_label.append('{}y'.format(int(e)))
    exp_idx = [exp_label.index(l) for l in ['1m','3m','1y','5y','10y','30y']]
    
    if axes is None:
        plt.boxplot(X,positions=range(0,len(expiries)),sym='.')
        if hline is not None:
            plt.plot([0,len(expiries)],[hline,hline],':k',linewidth=1)    
        if ylim is not None:
            plt.ylim(ylim)    
        plt.xticks(ticks=exp_idx,labels=[exp_label[i] for i in exp_idx])
        plt.title(title)
        plt.xlabel(xlabel)
    else:
        axes.boxplot(X,positions=range(0,len(expiries)),sym='.')
        if hline is not None:
            axes.plot([0,len(expiries)],[hline,hline],':k',linewidth=1)    
        if ylim is not None:
            axes.set_ylim(ylim)    
        axes.set_xticks(ticks=exp_idx)
        axes.set_xticklabels(labels=[exp_label[i] for i in exp_idx])
        axes.set_title(title)
        axes.set_xlabel(xlabel)    
        

    

def lds_recursions(X, A, G, C, S, mu0, V0):
    '''Forward and backward recursions for inference in LDS.
    We follow Bishop (2006) section 13.3 (+ 2011 errata).'''
        
    N,D = X.shape
    Q = len(mu0)
    mu = np.zeros((N,Q))
    V = np.zeros((N,Q,Q))
    K = np.zeros((N,Q,D))
    P = np.zeros((N,Q,Q))
    mu_hat = np.zeros((N,Q))
    V_hat = np.zeros((N,Q,Q))
    J = np.zeros((N,Q,Q))
    
    # Need to be careful with the indexing here, in Bishop n=1...N
    # but in Python, n = 0...N-1
    # Bishop section 13.3 denotes mu0 the (prior) mean of z1
    # Here mu[0] is the *posterior* mean of z1 given x1 i.e. mu1 in Bishop
    # and likewise, V[0] = V1 of Bishop
    # Formulas checked manually for V[0], K[0]
    # (the errata says to replace V0 by P0 but that's incorrect here)
    
    # Forward
    K[0] = V0 @ C.T @ np.linalg.inv(C @ V0 @ C.T + S)
    mu[0] = mu0 + K[0] @ (X[0] - C @ mu0)
    V[0] = (np.eye(Q) - K[0] @ C) @ V0
    
    for n in range(1,N):
        P[n-1] = A @ V[n-1] @ A.T + G
        K[n] = P[n-1] @ C.T @ np.linalg.inv(C @ P[n-1] @ C.T + S)
        mu[n] = A @ mu[n-1] + K[n] @ (X[n] - C @ A @ mu[n-1])
        V[n] = (np.eye(Q) - K[n] @ C) @ P[n-1]   
    
    # Backward
    mu_hat[-1] = mu[-1]
    V_hat[-1] = V[-1]    
    for n in range(N-2,-1,-1): # Need to stop at -1 not 0 !
        J[n] = V[n] @ A.T @ np.linalg.inv(P[n])
        mu_hat[n] = mu[n] + J[n] @ (mu_hat[n+1] - A @ mu[n]) # mu[n] not mu[N] (see Bishop errata)
        V_hat[n] = V[n] + J[n] @ (V[n+1] - P[n]) @ J[n].T        
        
    return mu, V, K, P, mu_hat, V_hat, J
    
    

    
    
########
class MPPCA:

    def __init__(self, M=1, Q=1, initialisation='gmm', max_iter=50, tol=1e-4):
        self.M = M
        self.Q = Q
        self.initialisation = initialisation
        self.max_iter = max_iter
        self.tol = tol        
        self.pi_ = None
        self.mu_ = None
        self.cov_ = None
        self.W_ = None
        self.sigma2_ = None
        self.R_ = None
        self.L_ = None
        self.sigma2hist_ = None
        self.P_ = None
        self.singular_values_ = None
        self.ev_ = None
    
    @property    
    def C_(self):
        '''Return model covariance as defined for PPCA: C = W W.T + s2 I'''
        C = []
        for m in range(self.M):
            C.append(self.W_[m] @ self.W_[m].T + self.sigma2_[m] * np.eye(self.W_[m].shape[0]))
        
        return np.array(C)
    
    def fit(self, X):
        
        N, D = X.shape
        M = self.M
        Q = self.Q
        
        # Initialise parameters
        if self.initialisation=='kmeans':    
            pi, mu, W, sigma2, clusters = mppca.initialization_kmeans(X, M, Q)
            
        elif self.initialisation=='random':            
            a = 10 # governs prob mass concentration for the Dirichlet distros
            pi = np.random.dirichlet(np.ones(M)*a)
            mu = np.random.randn(M,D)
            sigma2 = np.ones(M) * 1            
            W = np.array([np.sqrt(np.random.dirichlet(np.ones(Q)*a,D)) * np.sqrt(X.var(axis=0)-sigma2[m]).reshape(-1,1) for m in range(M)])
            # this definition ensures var(X) == observed var
            
        elif self.initialisation=='gmm':          
            gm = GaussianMixture(n_components=M, covariance_type='full')
            gm.fit(X)
            #resp_tr = gm.predict_proba(X_tr)
            #resp_te = gm.predict_proba(X_te)
            
            pi = gm.weights_.copy()
            mu = gm.means_.copy()
            sigma2 = np.full(M,np.nan)
            W = np.full((M,D,Q),np.nan)
            
            for m in range(M):
                C = gm.covariances_[0]
                ev, eV = eigh(C)
                sigma2[m] = np.mean(ev[Q:])
                dW = (np.random.rand()*2-1)*0.25
                W[m] = eV[:,:Q] @ np.power((ev[:Q]-sigma2[m])*np.eye(Q),1/2) * (1+dW)
            
        else:            
            assert(False, 'initialisation: random, kmeans or gmm')
        
        self.pi_, self.mu_, self.W_, self.sigma2_, self.R_, self.L_, self.sigma2hist_ = mppca.mppca_gem(X, pi, mu, W, sigma2, self.max_iter, self.tol)
        self.P_, ev = extract_pcs(self.W_, self.sigma2_)
        self.singular_values_ = np.sqrt(ev * N)
        self.ev_ = ev
        R = self.predict(X, output='r')
        cov = np.zeros((self.M,X.shape[1],X.shape[1]))
        for m in range(self.M):            
            cov[m] = ((R[:,[m]] * (X - self.mu_[m])).T @ (X - self.mu_[m])) / R[:,m].sum()        
        self.cov_ = cov
        
        
    def predict(self, X, output='m'):
        
        R = mppca.mppca_predict(X,self.pi_,self.mu_,self.W_,self.sigma2_)
        
        return R if output=='r' else np.argmax(R,axis=1)
        


######## 
class DynamicCenterer:

    def __init__(self):
        self.mean_ = []
        
    def fit(self, X):
        cs = np.cumsum(X,axis=0)
        N = np.arange(X.shape[0]).reshape(-1,1) + 1
        self.mean_ = cs / N
        return self
    
    def transform(self,X):
        return X - self.mean_
        
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self,X):
        return X + self.mean_
        
  
######### 
class ResidualsFitter:

    '''We model residuals as AR(1) processes: 
    R(t) = alpha * R(t-1) + epsilon(t)
    epsilon ~ N(0, epsilon_var) 
    '''
    def __init__(self, alpha=0.0001, **kwargs):
        self._reg = Ridge(alpha=alpha, **kwargs)
        
    def fit(self, R=None, sample_weight=None):
        '''Fit AR(1) model to each residuals series'''
        if R is not None:
            self.R = R if R.ndim==2 else R.reshape(-1,1) # assuming if not 2D it is 1D (and not 3D)
            self.alpha_ = np.full(self.D, np.nan)
            self.intercept_ = np.full(self.D, np.nan)
            self.alpha_se_ = np.full(self.D, np.nan)
            self.epsilon_var_ = np.full(self.D, np.nan)
            for d in range(self.D):
                x = self.R[:-1,[d]]
                y = self.R[1:,[d]]
                w = None if sample_weight is None else sample_weight[1:] #np.minimum(sample_weight[:-1], sample_weight[1:])
                #plt.plot(w)
                # Keep only obs where both x and y are valid
                idx_keep = ~np.isnan(x) & ~np.isnan(y)
                x = x[idx_keep].reshape(-1,1)
                y = y[idx_keep].reshape(-1,1)
                self.alpha_[d] = self._reg.fit(x, y, None if w is None else w).coef_[0]                
                self.intercept_[d] = self._reg.intercept_ if np.isscalar(self._reg.intercept_) else self._reg.intercept_[0]
                epsilon = self._reg.predict(x) - y
                self.epsilon_var_[d] = np.var(epsilon) # needed for error estimates around alpha
                sum_x2 = np.var(x) * self.N # denominator for SE(alpha)
                self.alpha_se_[d] = np.sqrt(self.epsilon_var_[d]/sum_x2) # SE(alpha)
                #R_serr[d] = np.sqrt(s2) # sd of epsilon
        else:
            warnings.warn('ResidualsFitter.fit() could not run: R is None!')     
            
        return self
                
    @property
    def N(self):
        return self.R.shape[0]

    @property
    def D(self):
        return self.R.shape[1]

    def expected_pnl(self, R=None, tau=1):
        '''Expected pnl based on mean-reversion strategy on residuals'''        
        R = self.R if R is None else R
        dev = self.intercept_/(1-self.alpha_) - R
        phi = np.sign(dev)
        return dev * phi * (1-self.alpha_.reshape(1,-1)**tau)
        
    def trading_stategy(self, R=None, m=1e-9):
        '''Simulate trading positions and pnl based on residuals'''       
        R = self.R if R is None else R
        
        dev = self.intercept_/(1-self.alpha_) - R
        phi = np.sign(dev)

        thr = m * self.epsilon_var_.reshape(1,-1) / 2 # Derived from concave utility model U(pnl) = (1-exp{-m*pnl}) / m
        # OR ...
        thr = m * np.sqrt(self.epsilon_var_.reshape(1,-1)) # Sharpe
        
        kc = 1 # TODO: Kelly criterion for sizing
        
        pos = phi * (self.expected_pnl(R,tau=1) > thr) * kc
        
        pnl = pos[:-1] * np.diff(R,axis=0) # TODO: add costs (when pos changes)
        pnl = np.append(np.zeros((1,self.D)),pnl,axis=0)        
        cum_pnl = np.cumsum(pnl,axis=0)
        pf_pnl = cum_pnl.sum(axis=1)
        
        return pf_pnl, cum_pnl, pnl, pos
    
    def predict(self,R):
        '''Predict mean reversion (not Rt+1)'''
        R_hat = -R[:-1] * (1-self.alpha_)
        return R_hat
        
        
class PPCA:
    
    # TODO: think more about what we do with the mean of X when it's not centered on the whole dataset (e.g. mixture models).    
    
    def __init__(self, X, mu=None, P=None, ev=None, s2=None, fit=True, orient_pc=False, copy=None, **kwargs):
        self.X = X
        if copy is None:
            self.mu = X.mean(axis=0) if mu is None else mu
            self.pca = None
            if fit:
                self.pca = PCA(n_components=self.D, **kwargs)
                self.pca.fit(self.X - self.mu)
                self.P = self.pca.components_.T
                self.ev = self.pca.singular_values_**2 / self.N
                self._s2 = None
            else:
                self.P = P
                self.ev = ev
                self._s2 = s2
            
            if orient_pc: # Ensure PCs have a positive first coordinate to facilitate comparisons across models
                signs = np.sign(self.P.sum(0)).reshape(1,-1)
                self.P = self.P * signs
            return None
        else: # To copy from another PPCA quickly
            self.mu = copy.mu
            self.P = copy.P
            self.ev = copy.ev
            self._s2 = copy._s2
    
    @property
    def Xbar(self):
        return self.X - self.mu
    
    @property
    def N(self):
        return self.X.shape[0]
        
    @property
    def D(self):
        return self.X.shape[1]
        
    @property
    def sv(self):
        return np.sqrt(self.ev * self.N)
    
    @property
    def var(self):
        '''Absolute variance coming from each factor for each series
        e.g. variance[k,f] is the variance of series (k) coming from factor (f)
        '''
        return self.P**2 * self.ev # NOT the variance of "returns" or "changes" but simply of observations (level)
    
    @property
    def vol(self):
        '''Absolute vol of each factor for each series'''
        return np.sqrt(self.var) # same comment as for 'var'
     
    @property
    def rvar(self):
        '''Relative variance of each series coming from each factor'''
        return self.var / self.var.sum(axis=1).reshape(-1,1)
    
    @property
    def rvol(self):
        '''Relative volatility of each series coming from each factor'''
        vol = np.sqrt(np.cumsum(self.var,1))
        dvol = np.diff(np.column_stack([np.zeros(self.D),vol]),axis=1)
        return dvol / vol[:,-1].reshape(-1,1)
        
    def P_surf(self,n_rows,n_cols):
        '''Return PCs arranged as 2D surface'''
        P = np.zeros((self.P.shape[1],n_rows,n_cols))
        for q in range(self.P.shape[1]):
            P[q] = self.P[:,q].reshape(n_rows,n_cols)
        return P
        
        
    def s2(self, Q=1):
        '''sigma2 in the PPCA model'''
        if (self._s2 is None) or (Q!=self.P.shape[1]):
            tot_var = self.X.var(axis=0)
            s2 = (tot_var.sum() - self.ev[:Q].sum()) / (self.D-Q) # Definition of sigma2
            #print(s2)
            return s2
        else:
            
            warnings.warn('s2 already set, returning set value!')
            return self._s2
     
    def W(self, Q=1):
        '''Get W in the PPCA sense'''        
        s2 = self.s2(Q=Q); #print(s2)
        L = (self.ev[:Q] - s2) * np.eye(Q) # Diagonal matrix of lambda - s2
        W = self.P[:,:Q] @ L**(1/2)
        return W    
    
    def ZQ(self, Q=1, normalise=False, projection='pca'):
        '''First Q factors series'''
        assert ((projection == 'pca') or (projection == 'ppca')), 'PCA or PPCA projections only!'
        m = np.sqrt(self.ev[:Q]) if normalise else 1
        if projection == 'pca':            
            return self.Xbar @ self.P[:,:Q] / m
            #return self.Xbar @ self.P[:,:Q] @ (1/self.ev[:Q]**0.5 * np.eye(Q))
        elif projection == 'ppca':
            W = self.W(Q=Q)
            Minv = np.linalg.inv(W.T @ W + self.s2(Q=Q)*np.eye(Q))            
            return self.Xbar @ W @ Minv.T * np.sqrt(self.ev[:Q]) / m
    
    def XQ(self, Q=1, projection='pca'):
        '''Reconstructed series with Q PCs'''
        
        if projection=='pca':
            Z = self.ZQ(Q=Q, projection=projection)
            return Z @ self.P[:,:Q].T + self.mu
        elif projection=='ppca':
            Z = self.ZQ(Q=Q, projection=projection,normalise=True)
            return Z @ self.WQ(Q=Q).T + self.mu            
    
    def RQ(self, Q=1, projection='pca'):
        '''Residuals for all series with Q PCs''' 
        #print('asdfa')
        return self.X - self.XQ(Q=Q, projection=projection)
    
    def s2Q(self, Q=1):
        '''Variance of isotropic Gaussian with Q PCs'''
        return self.ev[Q:].mean() if Q<self.D else 0
    
    def WQ(self,Q=1):
        '''Model W with Q PCs'''
        return self.P[:,:Q] @ np.diag(self.ev[:Q] - self.s2(Q=Q))**0.5
    
    def CQ(self, Q=1):
        '''Model covariance with Q PCs'''
        return self.WQ(Q=Q) @ self.WQ(Q=Q).T + self.s2Q(Q=Q) * np.eye(self.D)
        
    def splitQ(self, Q=1, k=0):
        '''Split series (k) into Q factors'''
        return self.ZQ(Q=Q,normalise=False) * self.P[k,:Q]    



class RVStrategy:

    def __init__(self, sh_min=0):
        self.sh_min = sh_min # Sharpe ratio hurdle        
        return None
        
    def run(self, R, kappa, std, c=0, tau_factor=np.sqrt(252)):
        # Decide positions to take
        exp_dR = np.abs(R * kappa.reshape(1,-1))
        exp_sh = (exp_dR - c) / std.reshape(1,-1) * tau_factor
        pos = (exp_sh > self.sh_min) * -np.sign(R)
        
        # Calc P&L
        dR = np.diff(R,axis=0)
        pnl = pos[:-1] * dR
        pnl = np.vstack([pnl, np.zeros(R.shape[1])]) # To keep shape the same
        # and indexing to the pnl array will return pnl aligned with the dataset
        # holding the point used to ENTER the trade
        # and realised pnl array will be aligned with expected pnl array
        
        return pnl, pos
        
    def run_multi(self, R, kappa, std, pi, c=0, tau_factor=np.sqrt(252)):
        # Extensions in case we have multiple dynamics to consider 
        # e.g. mixture model
        # We now assume R, kappa, std have an extra dimension 
        # corresponding to the number of dynamics M
        M = R.shape[0]
        
        exp_dR = np.zeros_like(R)
        exp_sh = np.zeros_like(R)
        pos = np.zeros_like(R)
        dR = np.zeros((R.shape[0],R.shape[1]-1,R.shape[2]))
        pnl = np.zeros_like(R)
        
        for m in range(M):
            # Decide positions to take
            exp_dR[m] = np.abs(R[m] * kappa[m].reshape(1,-1))
            exp_sh[m] = (exp_dR[m] - c) / std[m].reshape(1,-1) * tau_factor        
            pos[m] = (exp_sh[m] > self.sh_min) * -np.sign(R[m])
        
            # Calc P&L
            dR[m] = np.diff(R[m],axis=0)
            pnl[m][0:-1] = pos[m][:-1] * dR[m]
            #pnl[m] = np.vstack([pnl[m], np.zeros(R[m].shape[1])]) # To keep shape the same
            # and indexing to the pnl array will return pnl aligned with the dataset
            # holding the point used to ENTER the trade
            # and realised pnl array will be aligned with expected pnl array
            
        # Find the pi-weighted argmax sharpe at each point
        # This is the dynamics we are going to follow
        out_pnl, out_pos = np.zeros_like(pnl[0]), np.zeros_like(pos[0])
        idx_max = np.argmax(np.array(exp_sh)*np.array(pi).reshape(-1,1,1),axis=0) # weighted by pi #TODO: think about this here more
        # I think that's not goign to work well due to the observed conditionality
        # The MPPCA doesn't know the data is ordered and so it fits the pops where they are
        # In fact they are ordered and so it implies a transition proba as in HMMPPCA
        # We should synthetically create H and use it for decisions here
        # i.e. check what state we're in and use the transition probas for next state rather than pi
        # => we do that in run_multi_resp
        idx_c, idx_r = np.meshgrid(range(pnl[0].shape[1]),range(pnl[0].shape[0]))
        out_pnl = np.array(pnl)[idx_max,idx_r,idx_c]
        out_pos = np.array(pos)[idx_max,idx_r,idx_c]
        
        return out_pnl, out_pos
 
    def run_multi_resp(self, R, kappa, std, resp, H, c=0, tau_factor=np.sqrt(252)):
        # Extensions in case we have multiple dynamics to consider 
        # e.g. mixture model
        # We now assume R, kappa, std have an extra dimension 
        # corresponding to the number of dynamics M
        M = R.shape[0]
        
        exp_dR = np.zeros_like(R)
        exp_sh = np.zeros_like(R)
        pos = np.zeros_like(R)
        dR = np.zeros((R.shape[0],R.shape[1]-1,R.shape[2]))
        pnl = np.zeros_like(R)
        
        for m in range(M):
            # Decide positions to take
            exp_dR[m] = np.abs(R[m] * kappa[m].reshape(1,-1))
            exp_sh[m] = (exp_dR[m] - c) / std[m].reshape(1,-1) * tau_factor        
            pos[m] = (exp_sh[m] > self.sh_min) * -np.sign(R[m])
        
            # Calc P&L
            dR[m] = np.diff(R[m],axis=0)
            pnl[m][0:-1] = pos[m][:-1] * dR[m]
            #pnl[m] = np.vstack([pnl[m], np.zeros(R[m].shape[1])]) # To keep shape the same
            # and indexing to the pnl array will return pnl aligned with the dataset
            # holding the point used to ENTER the trade
            # and realised pnl array will be aligned with expected pnl array
            
        # Find the pi-weighted argmax sharpe at each point
        # This is the dynamics we are going to follow
        out_pnl, out_pos = np.zeros_like(pnl[0]), np.zeros_like(pos[0])
        idx_pi = (resp[:,0] < 0.5)*1 # 0 if pop 0, 1 if pop 1 (only works for 2 though)
        pi = H[idx_pi]
        selector = np.zeros_like(np.array(exp_sh))
        for m in range(M):
            selector[m] = np.array(exp_sh)[m] * pi[:,[m]]
        #idx_max = np.argmax(selector,axis=0); print(idx_max)
        idx_max = np.repeat(idx_pi.reshape(-1,1),pnl[0].shape[1],axis=1)#; print(idx_max) # TESTING THIS
        # weighted by pi #TODO: think about this here more        
        # I think that's not goign to work well due to the observed conditionality
        # The MPPCA doesn't know the data is ordered and so it fits the pops where they are
        # In fact they are ordered and so it implies a transition proba as in HMMPPCA
        # We should synthetically create H and use it for decisions here
        # i.e. check what state we're in and use the transition probas for next state rather than pi
        idx_c, idx_r = np.meshgrid(range(pnl[0].shape[1]),range(pnl[0].shape[0]))
        out_pnl = np.array(pnl)[idx_max,idx_r,idx_c]
        out_pos = np.array(pos)[idx_max,idx_r,idx_c]
        
        return out_pnl, out_pos 
        
class GaussImputer:

    def __init__(self, sample=False):
        self.sample = sample
        return None
        
    @property
    def std_(self):        
        return np.sqrt(self.var_)
    
    def fit(self, X):
        # Mean of available data
        self.mean_ = np.nanmean(X, axis=0)
        
        # Variance of available data
        self.var_ = np.nanvar(X, axis=0)        

        # Correlation where all data is simultaneously available
        D = X.shape[1]
        idx_all = (~np.isnan(X)).sum(1) == D
        self.corr_ = np.corrcoef(X[idx_all], rowvar=False)
        
        # Now compute covariance based on full pop var and complete obs corr
        # This ensures the cov matrix is PSD while still using all available data 
        # for marginal sufficient stats        
        self.cov_ = np.diag(self.std_) @ self.corr_ @ np.diag(self.std_)
        
        return self
        
    def transform(self, X):
    
        X_fill = X.copy()

        # mean, covariance
        mu = self.mean_        
        S = self.cov_

        # Let's do it row by row for now
        for r,x in enumerate(X):
            idx_out = np.isnan(x)
            if idx_out.sum() > 0: #TODO: add check for all out
                idx_in = ~idx_out
                x_in = x[idx_in]
                a = mu[idx_in]
                b = mu[idx_out]
                A = S[idx_in,:][:,idx_in]
                B = S[idx_out,:][:,idx_out]
                C = S[idx_in,:][:,idx_out]
                Ainv = np.linalg.inv(A)

                # TODO: extension, take a=mu[idx_in] measured on data preceding x only 
                # to recognise that the mean moves over time 
                # and the imputed series should see this effect i.e. their mean after imputation 
                # should be different from that measured only on avaiable data
                mu_out = b + C.T @ Ainv @ (x_in - a) 
                E_out = B - C.T @ Ainv @ C

                if self.sample:
                    x_fill = np.random.multivariate_normal(mu_out, E_out, tol=1e-4) # https://github.com/numpy/numpy/issues/10839
                else:
                    x_fill = np.random.multivariate_normal(mu_out, E_out * 0)

                X_fill[r,idx_out] = x_fill
        
        return X_fill
        
    def iterate(self, X, n_iter=10, rel_tol=1e-3, verbose=False):
        X_iter = X.copy()
        assert self.sample, 'Set sample=True to run iterative fitting!'                 
        self.cov_iter_ = np.full((n_iter,X.shape[1],X.shape[1]),np.nan)
        self.mean_iter_ = np.full((n_iter,X.shape[1]),np.nan)
        dcov, dmean = np.nan, np.nan
        for i in range(n_iter):            
            self.fit(X_iter) # refit on the imputed data
            X_iter = self.transform(X) # transform the base data (with missing obs)
            self.cov_iter_[i] = np.cov(X_iter,rowvar=False)
            self.mean_iter_[i] = X_iter.mean(axis=0)
            if i > 0: # Calc change in fitted params
                x, C_new, C_old = np.ones(X.shape[1]), self.cov_iter_[i], self.cov_iter_[i-1]
                v_new = x.reshape(1,-1) @ C_new @ x.reshape(-1,1)
                v_old = x.reshape(1,-1) @ C_old @ x.reshape(-1,1)
                dcov = np.abs(v_new / v_old - 1)[0][0]
                dmean = np.max(np.abs(self.mean_iter_[i] / self.mean_iter_[i-1] - 1))                                 
            if verbose:
                print('Gaussian imputer iteration', i, '(dmean={},dcov={})'.format(np.round(dmean,3),np.round(dcov,3)))
            if np.maximum(dcov,dmean) < rel_tol:
                print('Convergence criterion reached! Exiting at iteration',i)
                self.iter_ = i
                break
        
        self.fit(X_iter) # we must run one last fit after the last iteration
        
        return X_iter
            
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
   
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
