import numpy as np
from scipy.stats import truncnorm, norm
import Tasmanian
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numba import jit
import sys
from sys import exit
# import statsmodels.api as sm

class sgpc:
    """Python class for solving a Stochastic Growth Model
    with Irreversible Investments using Deterministic Parametrized Expectations 
    Approach (PEA). 
    
    { Min Weight Resids: Global, Finite Elements: Local}

    Note:
    
     The usage of the term GRID here is consistent with TASMANIAN and the 
     Sparse Grid literature's definition: It is a collection of sparse 
     coordinates (interpolation nodes) approximating multidimensional domains 
     of (unknown) functions, tensor basis functions, and, quadrature weights 
     for integration on those nodes.

    (c) 2020++, T. Kam (URL: phantomachine.github.io)"""

    def __init__(self, 
                e_min_scalelog=-3.0, 
                e_max_scalelog=3.0,        # e_min/max_scales shock s.d.
                K_min=None, 
                K_max=None,             # Override for custom bounds
                K_lbscale = 0.5, 
                K_ubscale=1.25,         # Bounds K rel.to steadystate
                α=0.33, 
                δ=0.025, 
                η=3.0, 
                logZ_ss=0.0, 
                ρ=0.8, 
                σ=0.05, 
                YKratio=0.11210762331838565,
                NGRID_Z = 3,
                MAXITER=500, 
                TOL=1e-4):

        # ------------------- Properties ----------------------------------

        # Smolyak Sparse Grid + Chebychev Interpolation Scheme
        # Or Leja sequential sparse grid rules + global or local Lagrange polys
        # Dimension of function input (domain)
        # self.iNumInputs = iNumInputs # using two inputs for testing

        # Dimension of function output (codomain)
        # self.NumOutput = NumOutput # 1D (real line)
        # self.NumOutput = NumOutput

        # Model parameters
        self.α = α   # Cobb-Douglas capital share
        self.δ = δ   # Capital depreciation rate
        self.YKratio = YKratio # Empirical US time-series average: Y/K
        self.β = 1.0/(self.α*YKratio + 1.0-self.δ)   # SDiscount factor
        self.η = η   # CRRA utility parameter
        self.ρ = ρ   # Persistence of skill endowment (AR1 process)
        self.σ = σ   # Volatility (std.dev.) of skill endowment shock
        self.logZ_ss = logZ_ss  # Unconditional mean (AR1 process)
        self.ε_mean, self.ε_var = truncnorm.stats(e_min_scalelog,
                                    e_max_scalelog, moments='mv')
        self.Zss = np.exp(((1.0-ρ)*logZ_ss + σ*self.ε_mean)/(1.0-ρ))

        # Bounds on the level of log-normal shock e
        self.e_min_scalelog = e_min_scalelog
        self.e_max_scalelog = e_max_scalelog
        self.e_min = np.exp(e_min_scalelog*self.σ)
        self.e_max = np.exp(e_max_scalelog*self.σ)
        self.bounds_shocks = np.array([[self.e_min, self.e_max],]) 

        # Bounds on the level of TFP (Z)
        z_extremum = lambda e: \
            ((1.0-self.ρ)*self.logZ_ss + np.log(e))/(1.0-self.ρ)
        self.Z_min = np.exp(z_extremum(self.e_min))
        self.Z_max = np.exp(z_extremum(self.e_max))
        self.bounds_exo = np.array([[self.Z_min, self.Z_max],]) # Note 2D array!
        self.NumOutput_exo = 1 # Dimension (co-domain)
        self.iNumInputs_exo = self.bounds_exo.shape[0] # Dimension (domain)

        # Endogenous parameters (steady state values)
        # From s.s. Euler, assuming Cobb-Douglas f(K,Z)
        self.Kss = self.SteadyState()["capital"]
        self.Yss = self.SteadyState()["output"]

        # Shift from [-1,1]^iNumInputs canonical to model's non-canonical domain
        if (K_min==None) or (K_min<=0.0):
            self.K_min = K_lbscale*self.Kss
        else:
            self.K_min = K_min

        # From s.s. capital law, assuming Cobb-Douglas f(K,Z)
        if (K_max==None) or (K_max<=0.0) or (K_max <= self.Kss):
            # C = 0 forever, Z = Z_max, K = K_max forever
            K_ub = (self.Z_max/self.δ)**(1.0/(1.0-self.α))
            self.K_max = min(K_ubscale*self.Kss, K_ub)
        else:
            self.K_max = K_max

        # Exogenous state space of Z (for quadrature)
        # σ is s.d. of normal distro of AR(1) shock for log(Z)
        self.σ_lognormal = (np.exp(σ**2.0)*(np.exp(σ**2.0)-1.0))**0.5

        # Joint state space (K,Z) (for interpolations)
        self.bounds = np.array([[self.K_min, self.K_max],
                                [self.Z_min, self.Z_max]])
        self.NumOutput = 1 # Dimension (co-domain)
        self.iNumInputs = self.bounds.shape[0] # Dimension (domain)

        # Markov chain approximation of AR(1) shocks
        self.seed = True # fix seeding random number simulation (alt: None)
        self.NGRID_Z = NGRID_Z
        S, self.P = self.ar2mc()
        self.S = np.exp(S)
    
        # Precision settings
        self.MAXITER_value = MAXITER
        self.TOL_value = TOL       # stopping criterion: operator iteration

        # Smoothing parameter(default=1.0)
        self.SMOOTH = 1.0

    ## ----- MODEL PRIMITIVES -------------------------------------
    def U(self, C):
        """Flow utility of consumption C"""
        if self.η == 1.0:
            u = np.log(C)
        elif self.η != 1.0 and self.η > 0.0:
            u = (C**(1-η) - 1.0)/(1-η)
        else:
            print("Oops! CRRA η must be strictly positive!")
        return u

    def U_prime(self, C):
        """Flow marginal utility of consumption"""
        if self.η == 1.0:
            uprime = 1/C
        elif self.η != 1.0 and self.η > 0.0:
            uprime = C**(-self.η)
        else:
            print("Oops! CRRA η must be strictly positive!")
        return uprime

    def U_prime_inverse(self, val):
        """Inverse marginal utility of consumption function.
        Given MU value, val, get the inverse: the consumption level
        c that attains val."""
        if self.η == 1.0:
            c = 1.0/val
        elif self.η != 1.0 and self.η > 0.0:
            c = val**(-1.0/self.η)
        else:
            print("Oops! CRRA η must be strictly positive!")
        return c

    def F(self, K, Z):
        """Total Resources at state (K,Z)"""
        return Z*K**(self.α) + (1-self.δ)*K

    def F_K(self, K, Z):
        """MPK - Total Resources at state (K,Z)"""
        return self.α*Z*K**(self.α -1.0) + (1-self.δ)

    def f(self, K, Z):
        """Output production at state (K,Z)"""
        return Z*K**(self.α)

    def SteadyState(self):
        """Non-stochastic steady-state equilibrium. Used as reference point for 
        solution space and also for simulations."""
                       
        # From s.s. Euler, assuming Cobb-Douglas f(K,Z)
        α, β, δ, Zss = self.α, self.β, self.δ, self.Zss
        Zss = self.Zss
        Kss = (α*β*Zss/(1.0-β*(1.0-δ)))**(1.0/(1.0-α))
        Yss = self.f(Kss, Zss)
        Css = self.F(Kss, Zss) - Kss
        Iss = max([self.f(Kss, Zss) - Css, 0.0])
        # Assume non-binding deterministic SS
        mss = 0.0

        out = {"technology"     : Zss,
                "capital"       : Kss, 
                "output"        : Yss,
                "consumption"   : Css, 
                "investment"    : Iss,
                "multiplier"    : mss,
              }
        return out

    ## ----- TOOLS ------------------------------------------------
    def supnorm(self, function1, function2):
        """Returns the absolute maximal (supremum-norm) distance between
        two arbitrary NumPy ND-arrays (function coordinates)"""
        return (np.abs(function1 - function2)).max()

    def logNormal_cdf(self, x, μ, σ):
        """
        Univariate log-Normal cumulative density function.

        Parameters:
        -----------
        x (float) the argument of the PDF. x > 0
        μ, σ (floats) parameters of the normal PDF of log values, 
             i.e., log(x). σ > 0

        Returns
        -------
        cdf (float, array), the value of the cumulative density function at each x.

        Depends on SciPy stats.norm.
        """
        if (x <= 0.0):
            cdf = 0.0
        else:
            logx = np.log(x)
            cdf = norm.cdf(logx, loc=μ, scale=σ)
        return cdf

    def logNormal_pdf (self, x, μ, σ):
        """
        Univariate log-Normal probability density function.

        Also known as the Cobb-Douglas PDF or the Anti log-Normal PDF. The Log 
        Normal PDF describes a random variable X whose logarithm, log(X), is 
        normally distributed.

        Parameters:
        -----------
        x (float) the argument of the PDF. x > 0
        μ, σ (floats) parameters of the normal PDF of log values, 
             i.e., log(x). σ > 0

        Returns
        -------
        pdf (float, array), the value of the cumulative density function at each x.
        """
        if (x <= 0.0):
            pdf = 0.0
        else:
            denominator = x*σ*np.sqrt(2.0*np.pi)
            pdf = np.exp(-0.5*((np.log(x) - μ)/σ)**2.0)/denominator
        return pdf

    def logNormalTruncated_pdf(self, x, μ, σ, a, b):
        """
        Univariate Truncated log-Normal probability density function. Support of this distribution is [a,b].

        Parameters:
        -----------
        x (float) the argument of the PDF. x > 0
        μ, σ (floats) parameters of the log normal PDF of log values, 
             i.e., log(x). σ > 0
        a, b (floats) the lower and upper truncation limits: a < x < b. 
             Note that a >= 0, since x > 0.

        Returns
        -------
        pdf (float, array), the value of the prob density function at each x.
        """
        # Check for illegal inputs
        if (self.σ <= 0.0):
            print("logNormalTruncated_pdf: Must have sigma > 0!")
            exit
        if (b <= a):
            print("logNormalTruncated_pdf: Illegal bounds. Must have b > a!")
            exit
        if (a < 0.0):
            print("logNormalTruncated_pdf: Illegal bounds. Must have a >= 0!")
            exit

        # Evaluate pdf
        if (x <= a) or (b <= x):
            pdf = 0.0
        else:
            lncdf_a = self.logNormal_cdf(a, μ, σ)
            lncdf_b = self.logNormal_cdf(b, μ, σ)
            lnpdf_x = self.logNormal_pdf(x, μ, σ)
            pdf = lnpdf_x / (lncdf_b - lncdf_a)
        return pdf

    def ar1_density_conditional(self, z):
        """
        AR(1) model for log(Z). Get pdf of Z_next conditional on knowing current Z. Default assumes that steady-state (unconditional mean) of
        log(Z) = 0. 

        Model is:
        Z_next = exp(logZ_ss)**(1-ρ) * (Z**ρ) * exp(σ*ε_next)
        ε_next ~ TruncNorm(0, 1)

        Parameters:
        -----------
        z (float) the level of *realized* random variable Z
        logZ_ss (float) the unconditional mean of log(Z)
        ρ, σ (floats) persistence and std deviation parameters. |ρ| < 1, σ > 0.

        Returns
        -------
        pdf_z (float, array), the value of the conditional prob density function of z_next, i.e, conditional on a given z.
        """
        # Mean of log(Z_next) process conditional on log(Z)
        μ = (1-self.ρ)*self.logZ_ss + self.ρ*np.log(z)
        # Bounds on Z_next - BOUNDS_EXO defined in 2D array format for TASMANIAN
        a, b = float(self.bounds_exo[:, 0]), float(self.bounds_exo[:, 1])
        # Assumes shocks are log-Normal so, conditional distro also log-Normal
        # Example: y := z_next
        pdf_znext = lambda y: self.logNormalTruncated_pdf(y, μ, self.σ, a, b)
        return pdf_znext

    def ar1(self, z, ε, log_level=False, shock_scale ="sd"):
        """
        Define AR(1) model for TFP shock process.
        """
        ρ = self.ρ
        σ = self.σ
        Zss = self.Zss
        if shock_scale == "unit":
            shock = ε
        elif shock_scale == "sd":
            shock = σ*ε
        else:
            # option to introduce float scaling
            shock = shock_scale*ε

        # Option to log-linear ar1 or exponential form:
        if log_level==False:
            znext = (Zss**(1.0-ρ))*(z**ρ)*np.exp(shock)
        else:
            znext = (1.0-ρ)*np.log(Zss) + ρ*z + shock
        return znext

    def ar2mc(self):
        """
        Approximate an AR1 model by a finite-state-space Markov Chain (MC)
        (Rouwenhorst 1995, Econometrica). This method outperforms earlier
        approximation schemes of Tauchen (1986) or Tauchen and Hussey (1991)
        when RHO is very close to 1: Kopecky and Suen (2010, RED).

        Input: AR(1) model parameters, y+ = RHO*y + SIGMA*u, u ~ Normal(0,1)
               N, desired cardinality of finite state space (Z) of approx. MC

        Output: (S, P), state space and Markov matrix
        """
        # Extract params from self object
        ρ, σ, N = self.ρ, self.σ, self.NGRID_Z
        # State space S
        bound = np.sqrt((N-1)/(1-ρ**2.0))*σ
        S = np.linspace(-bound, bound, N)
        # Transition probabilities (N = 2).
        p = (self.ρ + 1.0) / 2.0
        q = p
        # Initial P_temp is P for N = 2 case
        P_temp = np.array([[ p,  1-p ],
                           [ 1-q, q  ]])
        # Construct transition matrix P
        if N == 2:
            P = P_temp
        elif N > 2:
            # Recursive build of P for N > 2
            for n in range(3, N+1):
                block1 = np.zeros((n,n))
                block2 = block1.copy()
                block3 = block1.copy()
                block4 = block1.copy()
                # Fill with last iteration's P_temp
                block1[:-1,:-1] =  P_temp
                block2[:-1,1:] =  P_temp
                block3[1:,:-1] =  P_temp
                block4[1:,1:] =  P_temp
                # Update P_temp for next iteration
                P_temp = p*block1 + (1-p)*block2 + (1-q)*block3 + q*block4
                P_temp[1:-1, :] = P_temp[1:-1, :]/2
            # Final P for N > 2
            P = P_temp
        return S, P

    def SimulateMarkovChain(self, Z=None, P=None, mu=None, T=88888):
        """Simulate T-length observations of Markov chain (mu,P)"""
        """Note: Finite state space Z admit integers or reals"""
        if Z is None:
            Z = self.S
        if P is None:
            P = self.P
        if mu is None:
            if self.seed:
                # Fix a seeded random generator
                np.random.seed(52348282)
            # Define arbitrary initial uncond. distro over Z
            mu = np.random.rand(Z.size)
            mu = mu/mu.sum()
        data = np.empty(T)
        data[0] = np.random.choice(Z, replace=False, p = mu)
        for t in range(T-1):
            if self.seed:
                np.random.seed(t + 1234)
            # Find index/location of element in Z with value data[t]
            state = Z.tolist().index(data[t])
            # Given state index, draw new state from conditional distro
            data[t+1] = np.random.choice(Z, replace=False, p = P[state,:])
        return data

    def ErgodistMC(self, P):
        """Compute stationary distribution of an ergodic Markov Chain"""
        N_state = P.shape[0]
        z = np.zeros(N_state)
        # Normalization: right eigenvector (z) as prob. dist.
        z[-1] = 1.0
        # System of linear equations: find fixed point z
        PMI = P - np.eye(N_state)
        PMI[:,-1] = np.ones(N_state)
        lambda_inf = np.linalg.solve(PMI.T,z.T)
        return lambda_inf

    def StatusBar(self, iteration, iteration_max, stats1, width=15):
            percent = float(iteration)/iteration_max
            sys.stdout.write("\r")
            progress = ""
            for i in range(width):
                if i <= int(width * percent):
                    progress += "="
                else:
                    progress += "-"
            sys.stdout.write(
                "[ %s ] %.2f%% %i/%i, error = %0.10f    "
                % (progress,percent*100,iteration,iteration_max,stats1)
                )
            sys.stdout.flush()

    def MakeAllPolyGrid(self, Depth, Order, sRule="localp"):
        """Create sparse grid given parameters and interpolant type. Used for 
        interpolation of functions over joint state space containing both 
        *current* endogenous and exogenous states. See MakeExoPolyGrid() for 
        grid subspace defined over exogenous states only (used for integration 
        w.r.t. distribution of exogenous states). 
        
        Dependencies: Uses TASMANIAN.

        Parameters:
        -----------
        iNumInputs (int),   Dimension of function input (domain)
        NumOutput  (int),   Dimension of function output (codomain)
        Depth (int),        Non-negative integer controls the density of grid 
                            The initial 
                            construction of the local grids uses tensor 
                            selection equivalent to TasGrid::type_level
                            Depth is the L parameter in the formula in TASMANIAN
                            manual; i.e., the "level" in Smolyak. 
        Order (int),        Integer no smaller than -1.
                            1 : indicates the use of constant and linear
                                functions only
                            2 : would allow quadratics (if enough points are 
                                present)
                           -1 : indicates using the largest possible order for 
                                each point.
        
        sRule (str),        Choose from { rule_localp, 
                                          rule_semilocalp,
                                          rule_localp0,
                                          rule_localpb     }

        Returns
        -------
        grid (obj),         Grid scheme: domain, quadrature weights, etc. See 
                            TASMANIAN manual.
        """
        
        # Step 1. Define sparse grid and local poly rule
        grid = Tasmanian.makeLocalPolynomialGrid(self.iNumInputs, 
                                                    self.NumOutput,
                                                        Depth, Order, sRule)

        # Step 2. Transform to non-canonical domain.
        # self.bounds is np array of domain bounds
        # e.g., np.array([[self.K_min, self.K_max], [self.Z_min, self.Z_max]])
        grid.setDomainTransform(self.bounds)

        return grid

    def MakeExoPolyGrid(self, Depth, Order, sRule="localp"):
        """Like MakeAllPolyGrid() this method defines only sparse grid
        over exogenous random variables' state space. Used for computing 
        conditional expectations (integrals) of functions of endo- and 
        exo-genous states, over subspace of *exogenous*, *future* random 
        variables. 
        
        Dependencies: Uses TASMANIAN.
        
        Parameters:
        -----------
        iNumInputs (int),   Dimension of function input (domain)
        NumOutput  (int),   Dimension of function output (codomain)
        Depth (int),        Non-negative integer controls the density of grid 
                            The initial 
                            construction of the local grids uses tensor 
                            selection equivalent to TasGrid::type_level
                            Depth is the L parameter in the formula in TASMANIAN
                            manual; i.e., the "level" in Smolyak. 
        Order (int),        Integer no smaller than -1.
                            1 : indicates the use of constant and linear
                                functions only
                            2 : would allow quadratics (if enough points are 
                                present)
                           -1 : indicates using the largest possible order for 
                                each point.
        
        sRule (str),        Choose from { rule_localp, 
                                          rule_semilocalp,
                                          rule_localp0,
                                          rule_localpb     }

        Returns
        -------
        grid_exo (obj),     Grid scheme: domain, quadrature weights, etc. See 
                            TASMANIAN manual.
        """

        # Step 1. Define sparse grid and local poly rule
        grid_exo = Tasmanian.makeLocalPolynomialGrid(self.iNumInputs_exo, 
                                                        self.NumOutput_exo,
                                                          Depth, Order, sRule)

        # Step 2. Transform to non-canonical domain.
        # self.bounds is np array of domain bounds for exog vars
        # e.g., np.array([[self.Z_min, self.Z_max]])
        grid_exo.setDomainTransform(self.bounds_exo)

        return grid_exo

    def InterpSparse(self, grid_all, Y, Xi):
        """Given raw data points Y use local, piecewise polynomials for 
        Lagrange Interpolation over localized sparse grid. Then calculate batch interpolated values for new data Yi outside of the grid. GRID_ALL is obtained from MakeAllPolyGrid(). 

        Parameters:
        -----------
        grid_all (obj),    sparse grid object (TASMANIAN). grid_all.getPoints()
                           give (Npoints x Nvars) sparse grid domain of 
                           future exogenous states
        Y (array),        (Npoints x Nvars) Function values (data) defined on 
                           grid_all 
        Xi (array),       (Npoints_i x Nvars) coordinates on domain not 
                           defined in grid_all (to be interpolated over)

        Returns
        -------
        Yi (array),        (Npoints_i x Nvars) interpolated values on Xi 
        """

        # Step 1. Lagrange Interpolation to define interpolant over supplied needed data points Y defined over elements in X := grid.getPoints()
        grid_all.loadNeededPoints(Y)

        # Step 2. Interpolate values get Yi values over Xi points not on grid X
        # Data must be np.array of size (N_obs, iNumInputs) where 
        # iNumInputs = dim(Domain) = bounds.shape[0]
        Yi = grid_all.evaluateBatch(Xi)

        return Yi

    def PEA_deterministic(self, grid_all, grid_exo, efun_old):
        """Take as given data for expectation function evaluated at sparse 
        GRID_ALL, efun_old. Evaluate the Euler operator once to get update of 
        this as efun_new. This evaluation over grid points (K,Z) involves: 
            (1) Checking for KKT complementary slackness condition(s)
            (2) Quadrature integral approximation to construct efun_new
        Implements a version of Christiano and Fisher's non-stochastic PEA 
        method. Benefit is that we don't need to do costly nonlinear 
        Newton-Raphson type local root solvers. (This is a special feature of 
        this model though!)

        This is also where most custom model definition is specified, in terms 
        of its Recursive Equilibrium (Euler, Constraints) conditions. 

        Note to selves: 
        ~~~~~~~~~~~~~~~
        Future devs should make this as model-free as possible, in the style of 
        Miranda and Fackler's CompEcon MATLAB toolbox or the incomplete DOLO!

        Parameters:
        -----------
        grid_all (obj),    sparse grid object (TASMANIAN). grid_all.getPoints()
                           give (Npoints x Nvars) sparse grid domain of 
                           future exogenous states
        grid_exo (obj),    sparse grid object (TASMANIAN). grid_exo.getPoints()
                           give (Npoints_exo x Nvars_exo) sparse grid domain of 
                           future exogenous states
        efun_old (array),  (Npoints x 1) Expectation Function values on 
                           grid_all 

        Returns
        -------
        efun_new (array), (Npoints x 1) updated Expectation Function values 
                          on grid_all 
        """

        # Note: grid_exo is "pre-computed" outside of PEA iteration!
        Z_next_grid = grid_exo.getPoints()
        zWeights = grid_exo.getQuadratureWeights()
        w = np.atleast_2d(zWeights) # Special case 1D shock, convert to 
                                    # 2D array. Keep TASMANIAN happy!

        # Array C is of dimensionality: grid_all.shape
        X = grid_all.getPoints()
        C = np.zeros(X.shape[0])
        mu = C.copy()

        # Pre-allocate - we'll update this at end of loop (in Step 4)
        efun_new = efun_old.copy()

        # print(efun_old.shape)

        # Loop over all current states (K,Z) in grid_all
        for idx_state, state in enumerate(X):

            # Current state (K,Z) :=: idx_state
            k, z = state[0], state[1]

            # STEP 1.
            # Guess of current C(K,Z) given efun_old(K,Z)
            Efun_state = efun_old[idx_state]
            c = self.U_prime_inverse(Efun_state)
            # Ensure c > 0:
            c = np.maximum(c, 1e-12)
            C[idx_state] = c

            # STEP 2.
            # Evaluate continuation K_next(Z,K) enforced by C(K,Z)=c:   
            knext = self.F(k, z) - c       
            # Implied investment flow 
            y = self.f(k, z)
            invest = y - c
            # KKT complementary slackness check at state (K,Z)
            if (invest < 0.0):
                # Update with corner solution
                knext = (1.0-self.δ)*k
                # K_next[idx_state] = knext
                C[idx_state] = y
                mu[idx_state] = self.U_prime(y) - Efun_state
            
            # Force knext to be within bounds - else interpolation problem!
            k_ubcheck = np.minimum(knext, self.K_max)
            # # print(k_ubcheck)
            knext = np.maximum(k_ubcheck, self.K_min)
            # if knext > self.K_max:
            #     knext = self.K_max

            # if knext < self.K_min:
            #     knext = self.K_min

            # STEP 3. - Interpolate efun_old, Evaluate RHS integrand components

            # Conditional pdf of z_next **given** current Z = z:
            # WARNING: Currently support only truncnorm option!
            # (LAMBDA function): Evaluate Qz at any point z_next as: Qz(z_next)
            Qz = self.ar1_density_conditional(z)

            Intg = np.zeros(Z_next_grid.shape)
            
            for idx_znext, znext in enumerate(Z_next_grid):
                # Interpolation
                Xi = np.zeros((1,2))
                Xi[:,0], Xi[:,1] = [knext, znext]
                Efun_nextval = self.InterpSparse(grid_all, efun_old, Xi)
                # Guess of next C(K',Z') given efun_old(K',Z')
                cnext = self.U_prime_inverse(Efun_nextval)
                # Continuation state and investment
                inext = self.f(knext, znext) - cnext
                munext = 0.0
                # KKT check
                if (inext < 0.0):
                    cnext = self.f(knext, znext)
                    munext = self.U_prime(cnext) - Efun_nextval
                # Define integrand
                gval = self.U_prime(cnext)*self.F_K(knext, znext) \
                                                        - (1.0-self.δ)*munext
                Intg[idx_znext] = self.β*gval*Qz(znext)
            
            # STEP 4.
            # Evaluate integral using quadrature scheme on grid_exo 
            Integrand = Intg*w.T
            # RHS_Euler = np.sum(Integrand), update expectation fn guess
            efun_new[idx_state] = np.sum(Integrand)

        return efun_new, C, mu

    def Solve_PEA_TimeIteration(self, grid_all, grid_exo, 
                                                efun_old=None, DISPLAY=True):
        """Start with an initial guess of the expectation function (typically 
        the RHS of your Euler equation), efun_old. Iterate on a version of the 
        (Wilbur Coleman III)-cum-(Kevin Reffett) operator in terms of a 
        PEA_deterministic() operator defined in this same class, until 
        successive approximations of the equilibrium expectations function 
        converge to a limit. This allow you to then back out the solution's 
        implied equilibrium policy function(s).

        Note to selves: 
        ~~~~~~~~~~~~~~~
        Next-phase development: Endogenize grid_all to exploit adaptive sparse 
        grid capabilities starting from the most dense grid_all. Idea: as we 
        iterate closer to a solution, there may be redundant grid elements (in 
        terms of a surplus criterion). Then we might consider updating grid_all 
        with successively sparser grid_all refinements.

        Parameters:
        -----------
        Let X := grid_all.getPoints() from TASMANIAN.

        grid_all (obj),    sparse grid object (TASMANIAN). grid_all.getPoints()
                           give (Npoints x Nvars) sparse grid domain of 
                           future exogenous states
        grid_exo (obj),    sparse grid object (TASMANIAN). grid_exo.getPoints()
                           give (Npoints_exo x Nvars_exo) sparse grid domain of 
                           future exogenous states
        efun_old (array), (Npoints x Nvars) Expectation Function values on 
                          grid_all 

        Returns
        -------
        efun_new (array), (Npoints x 1) fixed-point Expectation Function 
                          values on grid_all 
        C (array),        (Npoints x 1) fixed-point policy (consumption) 
                          values on grid_all 
        mu (array),       (Npoints x 1) fixed-point multiplier function 
                          values on grid_all (for irreversible investment 
                          constraint)
        """
        if DISPLAY==True:
            print("\n****** Solve_PEA_TimeIteration ***********************\n")

            print("For function interpolations")
            print("----------------------------------")
            print("\tRule: %s" % grid_all.getRule())
            print("\tInterpolation Nodes: %i" % grid_all.getNumPoints()) 
            print("\tLocal Polynomial basis (Finite-Element Method)? %s" 
                                        % grid_all.isLocalPolynomial())
            print("\tGlobal Polynomial basis (Weighted Residuals Method)? %s" 
                                        % grid_all.isGlobal())
            print("\tInterpolation Nodes: %i" % grid_all.getNumPoints())
            print("\tMax. order of polynomials: %i" % grid_all.getOrder())

            print("\nFor quadrature/integration")
            print("----------------------------------")
            print("\tRule: %s" % grid_exo.getRule())
            print("\tInterpolation Nodes: %i" % grid_exo.getNumPoints()) 
            print("\tLocal Polynomial basis (Finite-Element Method)? %s" 
                                        % grid_exo.isLocalPolynomial())
            print("\tGlobal Polynomial basis (Weighted Residuals Method)? %s" 
                                        % grid_exo.isGlobal())
            print("\tInterpolation Nodes: %i" % grid_exo.getNumPoints())
            print("\tMax. order of polynomials: %i" % grid_exo.getOrder())

            print("\n\t\t請稍微等一下 ...")
            print("\t\tதயவுசெய்து ஒரு கணம் காத்திருங்கள் ...")
            print("\t\tしばらくお待ちください ...")
            print("\t\tPlease wait ...\n")

        if efun_old.all() == None:
            # Initial C guess is ad-hoc fraction of output
            K_grid = grid_all.getPoints()[:,0]
            Z_grid = grid_all.getPoints()[:,1]
            mpc = 0.21
            Kss = self.SteadyState()[0]
            K_grid[:] = Kss
            Z_grid[:] = self.Zss
            # Initial guess: Convert into column 2D Numpy array for TASMANIAN!
            efun_old = np.atleast_2d(mpc*self.F(K_grid, Z_grid)).T

        for j in range(self.MAXITER_value): 
            # Evaluate Euler operator once given efun_old
            efun_new, C, mu = \
                self.PEA_deterministic(grid_all, grid_exo, efun_old)
            # Compute distance between guesses
            error = self.supnorm(efun_new, efun_old)
            # Update expectation function
            efun_old = self.SMOOTH*efun_new + (1.0-self.SMOOTH)*efun_old
            # Progress bar
            self.StatusBar(j+1, self.MAXITER_value, error, width=15)

            # Stopping rules
            if (j == self.MAXITER_value-1) and (error >= self.TOL_value):
                print("\nSolve_PEA_TimeIteration: Max interation reached.")
                print("\t\tYou have not converged below tolerance!")
            if error < self.TOL_value:
                print("\nSolve_PEA_TimeIteration: Convergence w.r.t. TOL_value attained.")
                break

        return efun_new, C, mu

    def getInitialGuess(self, grid_all):
        # STEP 3: Initial guess of expectation function
        #
        # Homotopy?
        # Initial guess based on MU(C) from analytical textbook sgrowth model
        b0, b1, b2 = (1.0-self.α*self.β), self.α, 1.0
        efguess = lambda k, z: \
            self.U_prime( np.exp(b0 + b1*np.log(k) + b2*np.log(z)) )
        e0 = efguess(grid_all.getPoints()[:,0], grid_all.getPoints()[:,1])
        efun_init = np.atleast_2d(e0).T
        return efun_init

    def getPolicy(self, grid_all, function_array):
        """Construct approximant representations of a given function 
        data, function_array.
        
        Parameters:
        -----------
        Let X := grid_all.getPoints() from TASMANIAN.

        fun_array (array), (Npoints x 1) fixed-point Expectation Function 
                            values on X

        Returns
        -------
        function (obj),   interpolant object as defined
                          through wrapper function InterpSparse() above.
                          function = ["expectation", "consumption", "multiplier"]
        """

        # Ensure 2D numpy.ndarray format for TASMANIAN's evaluateBatch()
        function_array = np.atleast_2d(function_array).T
        # Define Lambda function objects
        function = lambda x: self.InterpSparse(grid_all, function_array, x)
        return function

    def showPolicy(self, grid_all, function_array, plt,
                    states_string=[r"$K$", r"$Z$"],           
                                    title_string=" ", PLOT=True):
        """Plot policy function given the sparse representation as 
        function_array defined on sparse grids grid_all.getPoints()"""

        # Sparse grid domain nodes
        gridPoints = grid_all.getPoints()

        # Define Lambda function objects via InterpSparse()
        policy = self.getPolicy(grid_all, function_array)

        # Uniform rectangular grid for plotting
        iTestGridSize = 50
        dX = np.linspace(self.K_min, self.K_max, iTestGridSize)  
        dY = np.linspace(self.Z_min, self.Z_max, iTestGridSize)
        # Create mesh-grid for MATPLOTLIB
        aMeshX, aMeshY = np.meshgrid(dX, dY)
        aTestPoints = np.column_stack([
                                      aMeshX.reshape((iTestGridSize**2, 1)),
                                      aMeshY.reshape((iTestGridSize**2, 1))
                                      ]) 
        Zval = policy(aTestPoints)
        Xmat = aTestPoints[:,0].reshape((iTestGridSize, iTestGridSize))
        Ymat = aTestPoints[:,1].reshape((iTestGridSize, iTestGridSize))
        Zmat = Zval.reshape((iTestGridSize, iTestGridSize))

        if PLOT==True:
            fig = plt.figure(facecolor="white")
            ax = plt.axes(projection='3d') 
            ax.grid("True")
            # Plot the Smolyak grid and interpolant on grid 
            ax.scatter3D(gridPoints[:,0], gridPoints[:,1], function_array, 
                                                                    alpha=1.0)

            # Surface plot of interpolated values
            surf = ax.plot_surface(Xmat, Ymat, Zmat, cmap=cm.coolwarm,
                                            linewidth=0.0, 
                                            antialiased=False,
                                            alpha=0.3)

            # Add a color bar which maps values to colors. 
            fig.colorbar(surf, shrink=0.5, aspect=5) 

            # Econ101 label yer bloody axes!
            plt.xlabel(states_string[0])
            plt.ylabel(states_string[1])
            plt.title(title_string)
            plt.show(block=False)
        # else:
        #     plt = []

        output = {  "xmat" : Xmat,
                    "ymat" : Ymat,
                    "zmat" : Zmat,
                    "plt"  : plt,
                }
        return output

    def showPolicyBatch(self, grid_all, efun, C, mu, PLOT=True):
        efun_1d = efun.flatten()          # convert back to 1d array
        results = [efun_1d.T, C, mu]      # Pack these arrays into a list
        titles = [  r"Expectation, $e(K,Z)$",
                    r"Consumption, $C(K,Z)$",
                    r"Multiplier, $\mu(K,Z)$"
                ]
        results_store = []              # For the data hoarder ...
        for idx_func, func in enumerate(results):
            # ieX = self.getPolicy(grid_all, efun)
            out_func = self.showPolicy(grid_all, func, plt,
                                        title_string=titles[idx_func], PLOT=PLOT)
            results_store.append(out_func)
            # if PLOT==True:
            #     out_func["plt"].show(block=False)

        return results_store

    def diagnostics_EulerError(self, grid_all, efun, C, mu,
                                method="informal", PLOT=True):
        """Given numerical solution for policy C and expectation function efun,
        Calculate the Den Haan and Marcet Euler equation errors.
        There are two methods:
        1.  DHM informal method - evaluates error between LHS and RHS of 
            Euler equation over a fine grid.
        2.  DHM formal method - This is akin to a GMM orthogonality conditions. 
            Idea: 
            * Simulate artificial time series of length T. 
            * Compute test statistic: J(T) := T*M'*inv(W)*M, where 
                M = (h(s)f(x, y, y')).sum()/T   # sum over length-T outcomes
                                                # h(s) can be 1, or any vector 
                                                # of variables.
                W = (f(x, y, y')h(s)'h(s)f(x, y, y')).sum()/T
              Test stat J(T) is distributed as χ**2 with |h(s)| degree of 
              freedom
        3. Repeat step 2, N independent times.
        4. Calculate the fraction of times the stat is in the lower- and 
           upper-5% range.
        """
        # policy = self.getPolicy(efun_new, C, mu)
        print("diagnostics_EulerError:")
        if method=="informal":
            
            print("\tCurrently performing 'informal' error diagnostic.")
            print("\tEuler equation error (percentage consumption changes).")
            # Sparse grid domain nodes
            gridPoints = grid_all.getPoints()

            # Define Lambda function objects via InterpSparse()
            C_interp = self.getPolicy(grid_all, C)
            mu_interp = self.getPolicy(grid_all, mu)
            e_interp = self.getPolicy(grid_all, (efun.flatten()).T)

            # Uniform rectangular grid for plotting
            iTestGridSize = 1000
            dX = np.linspace(self.K_min, self.K_max, iTestGridSize)
            dY = np.linspace(self.Z_min, self.Z_max, iTestGridSize)
            # Create mesh-grid for MATPLOTLIB
            aMeshX, aMeshY = np.meshgrid(dX, dY)
            aTestPoints = np.column_stack([
                                        aMeshX.reshape((iTestGridSize**2, 1)),
                                        aMeshY.reshape((iTestGridSize**2, 1))
                                        ])
            # Approximant evaluated over fine mesh
            C_approxval = C_interp(aTestPoints)
            mu_approxval = mu_interp(aTestPoints)
            # Implied consumption function evaluated over same mesh
            C_implied = self.U_prime_inverse(
                                    e_interp(aTestPoints) + mu_approxval
                                    )
            # Percentage errors in consumption errors
            error_pcons = np.abs((C_approxval - C_implied)/C_implied)
            error_pcons_max = error_pcons.max()
            error_pcons_avg = error_pcons.mean()

            print("\tMax. Euler (consumption) error = %6.5f percent" % \
                                                    (error_pcons_max*100))
            print("\tMean Euler (consumption) error = %6.5f percent" % \
                                                    (error_pcons_avg*100))

            # Plot figure of errors
            if PLOT==True:
                Xmat = aTestPoints[:, 0].reshape((iTestGridSize, iTestGridSize))
                Ymat = aTestPoints[:, 1].reshape((iTestGridSize, iTestGridSize))
                Zmat = error_pcons.reshape((iTestGridSize, iTestGridSize))

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.grid("True")
                # Surface plot of interpolated values
                surf = ax.plot_surface(Xmat, Ymat, Zmat, cmap=cm.coolwarm,
                                    linewidth=0.0,
                                    antialiased=False,
                                    alpha=0.3)

                # Add a color bar which maps values to colors.
                fig.colorbar(surf, shrink=0.5, aspect=5)

                # Econ101 label yer bloody axes!
                plt.xlabel(r"$K$")
                plt.ylabel(r"$Z$")
                plt.title("Euler Error")
                plt.show(block=False)
            # else:
            #     plt = []
        else:
            print("Currently performing Den-Haan and Marcet's J-test error diagnostic.")
            print("Sorry! This feature is not yet available in your region.")
        return [error_pcons_max, aTestPoints, error_pcons, error_pcons_avg, plt]

    def ImpulseResponseFunction(self, grid_all, C, mu,
                                Horizon=16, 
                                shock_scale="sd", 
                                shock_date=1,
                                experiment="deterministic", 
                                Burn=1000,
                                PLOT=True,
                                irf_percent=False,
                                show_det_steady=False,
                                Display=False,
                                state_init=[],
                                ):

        # Interpolant objects: C and mu functions
        C_interp = self.getPolicy(grid_all, C)
        mu_interp = self.getPolicy(grid_all, mu)
        # Initial state at steady state
        # state = np.atleast_2d([self.Kss, self.Zss])
        # kss, zss = state[:, 0].copy(), state[:, 1].copy()
        # Shock series
        if experiment=="deterministic":
            ε = np.zeros(Horizon)
            ε[shock_date] = 1.0
        else:
            e_min = np.log(self.e_min)
            e_max = np.log(self.e_max)
            ε = truncnorm.rvs(e_min, e_max, size=Burn+Horizon)

        # Certainty-equivalent steady states
        kss, zss = self.Kss, self.Zss
    
        # Initiate lists of simulated outcomes
        yss = self.SteadyState()["output"]
        css = self.SteadyState()["consumption"]
        xss = self.SteadyState()["investment"]
        mss = self.SteadyState()["multiplier"]
        zs, ks, ys, cs, xs, ms = [], [], [], [], [], []

        # Initialize state
        if not state_init:
            k, z = kss.copy(), zss.copy()
        else:
            k, z = state_init[0], state_init[1]
            # k, z = self.K_min, self.Z_min

        # Make at least 2D for TASMANIAN Evaluate() or EvaluateBatch()
        state = np.atleast_2d([k, z])
        
        # Recursively generate outcomes
        if experiment=="deterministic":
            T = Horizon
        else:
            T = Burn + Horizon
        
        for t in range(T-1):
            # Store current states (k,z)
            ks.append(k)
            zs.append(z)
            # pack into 2d Numpy array to suit interpolation
            # Current outcomes
            y = self.f(k,z)
            ys.append(y)      # output y
            
            # Current consumption
            if t == 0:
                c = css
            else:
                c = C_interp(state)
            # Current investment (could be < 0)
            x = y - c

            # check for KKT
            # if (x > 0.0):
            mu = mu_interp(state)
            if (mu > self.TOL_value) and (x < 0.0):
                knext = (1.0-self.δ)*k
                xs.append(0.0)                # Store x = 0
                ms.append(mu)                 # binding, mu > 0
                cs.append(y)                  # Store c = y
            else:
                knext = self.F(k,z) - c       
                xs.append(x)                  # Store x > 0
                ms.append(0.0)                # Store mu = 0
                cs.append(c)                  # Store c < y

            # Draw next-period TFP state, update states
            znext = self.ar1(z, ε[t+1], shock_scale=shock_scale)
            k, z = knext, znext
            state[:,0], state[:,1] = k, z
            # Report progress to screen (default=False)
            if Display:
                self.StatusBar(t+1, T-1, 0.0, width=15)
        
        # Flattened numpy array
        zs = np.asarray(zs).flatten()
        ks = np.asarray(ks).flatten()
        ys = np.asarray(ys).flatten()
        cs = np.asarray(cs).flatten()
        xs = np.asarray(xs).flatten()
        ms = np.asarray(ms).flatten()

        if irf_percent==True:
            zs = (zs - zss)/zss
            ks = (ks - kss)/kss
            ys = (ys - yss)/yss
            cs = (cs - css)/css
            xs = (xs - xss)/xss
            # ms = (ms - muss)/muss     # (not defined if muss=0!)

        # For time-series sim, remove initial burn-in obs
        if experiment != "deterministic":
            zs = zs[Burn::]
            ks = ks[Burn::]
            ys = ys[Burn::]
            cs = cs[Burn::]
            xs = xs[Burn::]
            ms = ms[Burn::]
            print("\n\tBurn-in sims. of length BURN=%i discarded ..." % Burn)

        # Pack away
        sims = {    
                    "technology"  : { "path" : zs, "point" : zss },
                    "capital"     : { "path" : ks, "point" : kss },
                    "output"      : { "path" : ys, "point" : yss },
                    "consumption" : { "path" : cs, "point" : css },
                    "investment"  : { "path" : xs, "point" : xss },
                    "multiplier"  : { "path" : ms, "point" : mss },
                }

        # Default option to PLOT time-series/IRF figures
        if PLOT == True:
            nvars = len(sims)
            ncol = 2
            nrow = int((nvars + np.mod(nvars, ncol))/ncol)

            fig = plt.figure(facecolor="white", tight_layout=True)
            for idx_key, (key, series) in enumerate(sims.items()):
                T_series = len(series["path"])
                plt.subplot(nrow, ncol, idx_key+1)
                #plt.subplots_adjust(hspace = .001)
                plt.plot(np.arange(T_series), series["path"], 'k.--')
                if (show_det_steady==True):
                    if (irf_percent):
                        plt.plot(np.arange(T_series), np.zeros(T_series), 'r')
                    else:
                        plt.plot(np.arange(T_series), \
                                        series["point"]*np.ones(T_series), 'r')
                
                ax = fig.gca()
                ax.set_ylabel(key)
                # ax.set_xlabel('t')
            plt.show(block=False)
        return sims



