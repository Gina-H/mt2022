from sgpc import sgpc as stgiri
import time
import numpy as np
# import matplotlib.pyplot as plt

def run(    α=0.3, 
            δ=0.019, 
            η=1.5, 
            ρ=0.90,         # 0.8,  0.85. 0.8999
            σ=0.05,         # 0.05, 0.05, 0.0072
            K_lbscale=0.25,  # 0.5, 
            K_ubscale=1.6,  # 1.25,
            e_min_scalelog=-2.0, #3.0 
                                    # Some issues if 3 for high (ρ, σ)!
            e_max_scalelog=2.0,  #3.0 # e_min/max_scales shock s.d.
            TOL=1e-8,
            L_all = 5,      # Current state-space: grid density (L)evel
            Order_all = 7,  #            Max. order of polynomial bases
            L_exo = 5,      # Future exo state-space: grid density (L)evel
            Order_exo = 1,  #            Max. order of polynomial bases
            LOAD_OLD = False, # Load previously saved exp.func. array?
            PLOT = True,     # Plot results?
            Display=True,
            # YKratio=0.2399, # override baseline - Simon's setting
            show_det_steady=True, # Plot det.SS path?
            irf_percent=False, # Plot IRF as %-deviation from det.SS?
        ):
    """MAIN() executes a parametric instance of the stochastic growth model with ocassionally binding irreversible (putty-clay) investment constraint.
    Parameters and settings have default values assign. Override/customize your experiments by manual override of these inputs.

    Experience: Keep L_all >= L_exo, Order_all >= Order_exo

    To execute:
        import main
        >> main.run()

    (c) 2020++ T. Kam (phantomachine.github.io)
    """

    
    # STEP 1: Instantiate class STGIRI as MODEL object
    model = stgiri( α=α, δ=δ, η=η, ρ=ρ, σ=σ, 
                    K_lbscale=K_lbscale, 
                    K_ubscale=K_ubscale,  
                    e_min_scalelog=e_min_scalelog,
                    e_max_scalelog=e_max_scalelog,  
                    TOL=TOL
                    )

    # STEP 2: Define sparse-grid schemes ...
    #
    # - for current (K,Z)-space/domain - interpolation
    
    grid_all = model.MakeAllPolyGrid(Depth=L_all, Order=Order_all, \
                                                        sRule="localp")

    # - for future (Z+)-space/domain - integration
    grid_exo = model.MakeExoPolyGrid(Depth=L_exo, Order=Order_exo, \
                                                        sRule="localp")

    # Name experiment (str)
    outfilename = "Depth-" + str(L_all) \
                + "-Order-" + str(Order_all) \
                + "-Shock-" + str(int(model.e_max_scalelog))
    dir = "out/"
    filename = dir + outfilename + '.npy'

    # STEP 3: Initial guess of expectation function
    if LOAD_OLD == False:
        # Homotopy? Initial guess based on MU(C) from
        # analytical textbook sgrowth model
        print("LOAD_OLD=False: Fresh initial guess ...")
        efun_init = model.getInitialGuess(grid_all)
    else:
        print("LOAD_OLD=True: Using old result as initial guess ...")
        print("Getting file from " + filename)
        efun_init = np.load(filename)

    # STEP 4: Solve
    tic = time.time()
    efun, C, mu = model.Solve_PEA_TimeIteration(grid_all, grid_exo,
                                                        efun_old=efun_init,
                                                        DISPLAY=Display)

    toc = time.time() - tic
    print("\n\nElapsed time:", toc, "seconds")

    print("\nNow saving results to .NPY file ...")
    np.save(filename, efun)

    # STEP 5: Get interpolants of solution functions and visualize
    results = model.showPolicyBatch(grid_all, efun, C, mu, PLOT=PLOT)

    # STEP 6: Calculate max Euler error and plot error surface
    diagnostics = model.diagnostics_EulerError(grid_all, efun, C, mu,
                                        method="informal", PLOT=PLOT)

    # STEP 7: Display impulse responses
    irf = model.ImpulseResponseFunction(grid_all, C, mu,
                                        Horizon=500,
                                        shock_scale="sd",
                                        experiment="deterministic",
                                        shock_date=1,
                                        PLOT=PLOT,
                                        irf_percent=irf_percent,
                                        show_det_steady=show_det_steady)

    # STEP 8: Generate one sample time-series path
    sims = model.ImpulseResponseFunction(grid_all, C, mu,
                                       Horizon=200,
                                       shock_scale="sd",
                                       experiment="stochastic",
                                       PLOT=PLOT,
                                       Burn=0,
                                       irf_percent=False)

    # STEP 9: Store results for posterity
    results_main = {    "policies"     : [efun, C, mu],
                        "grid_all"     : grid_all,
                        "grid_exo"     : grid_exo,
                        "results"      : results,
                        "diagnostics"  : diagnostics,
                        "irf"          : irf,
                        "time series"  : sims,
                   }

    # Output of MAIN()
    return results_main

