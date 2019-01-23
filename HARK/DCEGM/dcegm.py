# NOTE TO SELF: we manually place 0 (1e-6) in the grids, and that helps with
# interpolating on common grid

import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
from HARK import Solution, AgentType
from HARK.interpolation import LinearInterp, calcLogSumChoiceProbs
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv
from HARK.simulation import drawMeanOneLognormal
from math import sqrt
# might as well alias them utility, as we need CRRA
# should use these instead of hardcoded log (CRRA=1)
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityP_inv  = CRRAutilityP_inv
utility_inv   = CRRAutility_inv

RetiringDeatonParameters = namedtuple('RetiringDeatonParamters',
                                      'DiscFac CRRA DisUtil Rfree YRet YWork sigma')

def nonlinspace(low, high, n, phi = 1.1):
    '''
    Recursively constructs a non-linearly spaces grid that starts at low, ends at high, has n
    elements, and has smaller step lengths towards low if phi > 1 and smaller
    step lengths towards high if phi < 1.
    Taken from original implementation of G2EGM.
    Parameters
    ----------
    low : float
        first value in the returned grid
    high : float
        last value in the returned grid
    n : int
        number of elements in grid
    phi : float
        degree of non-linearity. Default is 1.5 (smaller steps towards `low`)
    Returns
    -------
    x : numpy.array
        the non-linearly spaced grid
    '''
    x    = numpy.ones(n)*numpy.nan
    x[0] = low
    for i in range(1,n):
        x[i] = x[i-1] + (high-x[i-1])/((n-i)**phi)
    return x

class RetiringDeatonSolution(Solution):
    def __init__(self, rs, ws, M, C, V_T, P):
        self.rs = rs
        self.ws = ws
        self.Coh = M
        self.C = C
        self.V_T = V_T
        self.P = P

class RetiredDeatonSolution(Solution):
    def __init__(self, Coh, C, CFunc, V_T, V_TFunc):
        self.Coh = Coh
        self.C = C
        self.CFunc = CFunc
        self.V_T = V_T
        self.V_TFunc = V_TFunc

class WorkingDeatonSolution(Solution):
    def __init__(self, Coh, C, CFunc, V_T, V_TFunc):
        self.Coh = Coh
        self.C = C
        self.CFunc = CFunc
        self.V_T = V_T
        self.V_TFunc = V_TFunc

class RetiringDeaton(AgentType):
    def __init__(self, DiscFac=0.98, Rfree=1.02, DisUtil=-1.0,
                 shiftPoints=False, T=20, CRRA=1.0, sigma=0.0,
                 YWork=20.0,
                 YRet=0.0, # normalized relative to work
                 TranIncVar = 0.005,
                 TranIncNodes = 0,
                 CohNodes=1900,
                 PdCohLims=(1e-6, 700), PdCohNodes=1800,
                 saveCommon=False,
                 **kwds):

        AgentType.__init__(self, **kwds)
        self.T = T
        self.simN = 100
        self.simT = 20
        self.saveCommon = saveCommon
        # check conditions for analytic solution
        # if not CRRA == 1.0:
        #     # err
        #     return None
        #     # Todo the DisUtil is wrong below, look up in paper
        # if DiscFac*Rfree >= 1.0 or -DisUtil > (1+DiscFac)*numpy.log(1+DiscFac):
        #     # err
        #     return None

        self.TranIncVar = TranIncVar
        self.TranIncNodes = TranIncNodes

        self.time_inv = ['PdCohGrid', 'CohGrid', 'EGMVector', 'par', 'Util', 'UtilP',
                         'UtilP_inv', 'saveCommon', 'TranInc', 'TranIncWeights']
        self.time_vary = ['age']

        self.age = list(range(T-1))

        self.par = RetiringDeatonParameters(DiscFac, CRRA, DisUtil, Rfree, YRet, YWork, sigma)
        self.PdCohLims = PdCohLims
        self.PdCohNodes = PdCohNodes
        self.CohNodes = CohNodes
        # d == 2 is working
        # - 10.0 moves curve down to improve lienar interpolation
        self.Util = lambda c, d: utility(c, CRRA) - self.par.DisUtil*(2-d) - 10.0
        self.UtilP = lambda c, d: utilityP(c, CRRA) # we require CRRA 1.0 for now...
        self.UtilP_inv = lambda u, d: utilityP_inv(u, CRRA) # ... so ...

        self.preSolve = self.updateLast
        self.solveOnePeriod = solveRetiringDeaton

    def updateLast(self):
        """
        Updates grids and functions according to the given model parameters, and
        solves the last period.

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.PdCohGrid = nonlinspace(self.PdCohLims[0], self.PdCohLims[1], self.PdCohNodes)
        self.CohGrid = nonlinspace(0, self.PdCohLims[1]*1.5, self.PdCohNodes)
        self.EGMVector = numpy.zeros(self.CohNodes)

        if self.TranIncNodes == 0:
            self.TranInc = numpy.ones(1)
            self.TranIncWeights = numpy.ones(1)
        elif self.TranIncNodes >= 1:
            self.TranInc, self.TranIncWeights = numpy.polynomial.hermite.hermgauss(self.TranIncNodes)
            self.TranInc = numpy.exp(-self.TranIncVar/2.0 + sqrt(2)*sqrt(self.TranIncVar)*self.TranInc)
            self.TranIncWeights = self.TranIncWeights/sqrt(numpy.pi)
            # self.TranInc = numpy.exp(-self.TranIncVar/2.0 + sqrt(2)*sqrt(self.TranIncVar)*self.TranInc)
        # else: # monte carlo
        #     self.TranInc = drawMeanOneLognormal(N=self.TranIncNodes, sigma=self.TranIncVar)
        #     self.TranIncWeights = numpy.ones(self.TranIncNodes)/self.TranIncNodes

        rs = self.solveLastRetired()
        ws = self.solveLastWorking()

        commonM = ws.Coh

        if self.saveCommon:
            # To save the pre-disrete choice expected consumption and value function,
            # we need to interpolate onto the same grid between the two. We know that
            # ws.C and ws.V_T are on the ws.Coh grid, so we use that to interpolate.
            Crs = LinearInterp(rs.Coh, rs.C)(commonM)
            Cws = ws.C

            V_Trs = LinearInterp(rs.Coh, rs.V_T)(commonM)
            V_Tws = ws.V_T

            # use vstack to stack the choice specific value functions by
            # value of cash-on-hand
            V_T, P = calcLogSumChoiceProbs(numpy.stack((V_Trs, V_Tws)), self.par.sigma)

            # find the expected consumption prior to discrete choice by element-
            # wise multiplication between choice distribution and choice specific
            # consumption functions
            C = (P*numpy.stack((Crs, Cws))).sum(axis=0)
        else:
            C, V_T, P = None, None, None

        self.solution_terminal = RetiringDeatonSolution(rs, ws, commonM, C, V_T, P)

    def solveLastRetired(self):
        """
        Solves the last period of a retired agent.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        choice = 1
        C = self.CohGrid # consume everything
        Coh = self.CohGrid # everywhere

        V_T = numpy.divide(-1.0, self.Util(self.CohGrid, choice))

        CFunc = lambda coh: LinearInterp(Coh, C, lower_extrap=True)(coh)
        V_TFunc = lambda coh: LinearInterp(Coh, V_T, lower_extrap=True)(coh)
        return RetiredDeatonSolution(Coh, C, CFunc, V_T, V_TFunc)

    def solveLastWorking(self):
        """
        Solves the last period of a working agent.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        choice = 2
        Coh = self.CohGrid # everywhere
        C = self.CohGrid # consume everything

        V_T = numpy.divide(-1.0, self.Util(self.CohGrid, choice))

        # Interpolants
        CFunc = lambda coh: LinearInterp(Coh, C)(coh)
        V_TFunc = lambda coh: LinearInterp(Coh, V_T)(coh)

        return WorkingDeatonSolution(Coh, C, CFunc, V_T, V_TFunc)

    # Simulation of RetiringDeatons
    def simulate(self):
        # need to set simulation parameters on init
        # verify that simT <= T
        CohData = numpy.zeros((self.simT, self.simN))
        CData = numpy.copy(CohData)
        PdCohData = numpy.copy(CohData)
        WorkData = numpy.copy(CohData)
        if self.TranIncNodes > 0:
            InkShkData = drawMeanOneLognormal(self.simN*self.simT, sigma=sqrt(self.TranIncVar)).reshape((self.simT, self.simN))
        else:
            InkShkData = numpy.ones(self.simN*self.simT)

        discreteThrsh = numpy.random.rand(self.simN, self.simT)
        for t in range(self.T-1):
            if t == 0:
                CohData[t] = self.par.YWork*InkShkData[t]
            else:
                CohData[t] = self.par.Rfree*PdCohData[t-1]+WorkData[t-1]*self.par.YWork*InkShkData[t]

            # determine discrete choice
            Vrs = self.solution[t].rs.V_TFunc(CohData[t]) # value of retiring
            Vws = self.solution[t].ws.V_TFunc(CohData[t]) # value of working

            V, P = calcLogSumChoiceProbs(numpy.stack((Vrs.T, Vws.T)), self.par.sigma)

            WorkData[t] = P[1] # this is a hack! does not work for positive sigma or multiple work levels

            CData[t][P[0]==1] = self.solution[t].rs.CFunc(CohData[t][P[0]==1])
            CData[t][P[1]==1] = self.solution[t].ws.CFunc(CohData[t][P[1]==1])
            # determine continuous choice
            PdCohData = CohData[t] - CData[t]


        self.CohData = CohData.T
        self.CData = CData.T
        self.PdCohData = PdCohData.T
        self.IncData = InkShkData.T
        self.WorkData = WorkData.T

    # Plotting methods for RetiringDeaton
    def plotV(self, t, d, plot_range = None):

        if t == self.T:
            sol = self.solution_terminal
        else:
            sol = self.solution[self.T-1-t]

        if d == 1:
            sol = sol.rs
            choice_str = "retired"
        elif d == 2:
            sol = sol.ws
            choice_str = "working"

        if plot_range == None:
            plot_range = range(len(sol.Coh))
        else:
            plot_range = range(plot_range[0], plot_range[1])
        plt.plot(sol.Coh[plot_range], numpy.divide(-1.0, sol.V_T[plot_range]), label="{} (t = {})".format(choice_str, t))
        plt.legend()
        plt.xlabel("Coh")
        plt.ylabel("V(Coh)")
        plt.title('Choice specific value functions')


    def plotC(self, t, d, plot_range = None, **kwds):

        if t == self.T:
            sol = self.solution_terminal
        else:
            sol = self.solution[self.T-1-t]

        if d == 1:
            sol = sol.rs
            choice_str = "retired"
        elif d == 2:
            sol = sol.ws
            choice_str = "working"

        if plot_range == None:
            plot_range = range(len(sol.Coh))
        else:
            plot_range = range(plot_range[0], plot_range[1])
        plt.plot(sol.Coh[plot_range], sol.C[plot_range], label="{} (t = {})".format(choice_str, t), **kwds)
        plt.legend()
        plt.xlabel("Coh")
        plt.ylabel("C(Coh)")
        plt.title('Choice specific consumption functions')


# Functions used to solve each period
def solveRetiringDeaton(solution_next, PdCohGrid, CohGrid, EGMVector, par, Util, UtilP, UtilP_inv, age, saveCommon, TranInc, TranIncWeights):
    """
    Solves a period of problem defined by the RetiringDeaton AgentType. It uses
    DCEGM to solve the mixed choice problem.

    Parameters
    ----------

    Returns
    -------

    """
    rs = solveRetiredDeaton(solution_next, PdCohGrid, EGMVector, par, Util, UtilP, UtilP_inv)
    ws = solveWorkingDeaton(solution_next, PdCohGrid, CohGrid, EGMVector, par, Util, UtilP, UtilP_inv, TranInc, TranIncWeights)

    if saveCommon:
        # To save the pre-disrete choice expected consumption and value function,
        # we need to interpolate onto the same grid between the two. We know that
        # ws.C and ws.V_T are on the ws.Coh grid, so we use that to interpolate.
        Crs = LinearInterp(rs.Coh, rs.C)(ws.Coh)
        Cws = ws.C
#        Cws = LinearInterp(rs.Coh, rs.C)(CohGrid)

        V_Trs = LinearInterp(rs.Coh, rs.V_T)(ws.Coh)
        V_Tws = ws.V_T

        V_T, P = calcLogSumChoiceProbs(numpy.stack((V_Trs, V_Tws)), par.sigma)

        C = (P*numpy.stack((Crs, Cws))).sum(axis=0)
    else:
        C, V_T, P = None, None, None

    return RetiringDeatonSolution(rs, ws, CohGrid, C, V_T, P)#-1.0/V, P) #0,0 is m and c

def solveRetiredDeaton(solution_next, PdCohGrid, EGMVector, par, Util, UtilP, UtilP_inv):
    choice = 1
    rs_tp1 = solution_next.rs
    Coh_t = numpy.copy(EGMVector)
    Cons_t = numpy.copy(EGMVector)
    EutilityP = numpy.copy(EGMVector)
    Ev = numpy.copy(EGMVector)

    aLen = len(PdCohGrid)
    conLen = len(Coh_t)-aLen

    # Next-period initial wealth given exogenous PdCohGrid
    Coh_tp1 = par.Rfree*PdCohGrid + par.YRet

    # Prepare variables for EGM step
    Cons_tp1 = rs_tp1.CFunc(Coh_tp1)
    V_T_tp1 = rs_tp1.V_TFunc(Coh_tp1)
    V_tp1 = numpy.divide(-1.0, V_T_tp1)


    # Calculate the expected marginal utility and expected value function
    EutilityP[conLen:] = par.Rfree*UtilP(Cons_tp1, choice)
    Ev[conLen:] = V_tp1

    # EGM step
    Cons_t[conLen:] = UtilP_inv(par.DiscFac*EutilityP[conLen:], choice)
    Coh_t[conLen:] = PdCohGrid + Cons_t[conLen:]

    # Add points to M (and C) to solve between 0 and the first point EGM finds
    # (that is add the solution in the segment where the agent is constrained)
    Coh_t[0:conLen] = numpy.linspace(0, Coh_t[conLen]*0.99, len(Coh_t)-aLen)
    Cons_t[0:conLen] = Coh_t[0:conLen]

    # Since a is the save everywhere (=0) the expected continuation value
    # is the same for all indeces 0:conLen
    Ev[0:conLen] = Ev[conLen+1]

    V_T = numpy.divide(-1.0, Util(Cons_t, choice) + par.DiscFac*Ev)

    CFunc = LinearInterp(Coh_t, Cons_t)
    V_TFunc = LinearInterp(Coh_t, V_T)

    return RetiredDeatonSolution(Coh_t, Cons_t, CFunc, V_T, V_TFunc)


def solveWorkingDeaton(solution_next, PdCohGrid, CohGrid, EGMVector, par, Util, UtilP, UtilP_inv, TranInc, TranIncWeights):
    choice = 2
    rs_tp1, ws_tp1 = solution_next.rs, solution_next.ws

    # Make vectors for Cash-on-hand (Coh), consumption (Cons_t)
    Coh_t = numpy.copy(EGMVector)
    Cons_t = numpy.copy(EGMVector)
    EutilityP = numpy.copy(EGMVector)
    Ev = numpy.copy(EGMVector)

    aLen = len(PdCohGrid)
    conLen = len(Coh_t) - aLen
    # Next-period initial wealth given exogenous PdCohGrid
    Cohrs_tp1 = par.Rfree*numpy.expand_dims(PdCohGrid, axis=1) + par.YRet*TranInc.T
    Cohws_tp1 = par.Rfree*numpy.expand_dims(PdCohGrid, axis=1) + par.YWork*TranInc.T

    # Prepare variables for EGM step
    Cr_tp1 = rs_tp1.CFunc(Cohrs_tp1)
    Cw_tp1 = ws_tp1.CFunc(Cohws_tp1)
    Vr_T = rs_tp1.V_TFunc(Cohrs_tp1)
    Vw_T = ws_tp1.V_TFunc(Cohws_tp1)

    # Due to the transformation on V being monotonic increasing, we can just as
    # well use the transformed values to do this discrete envelope step.
    V_T, ChoiceProb_tp1 = calcLogSumChoiceProbs(numpy.stack((Vr_T, Vw_T)), par.sigma)

    # Calculate the expected marginal utility and expected value function
    EutilityP[conLen:] =  par.Rfree*numpy.dot((ChoiceProb_tp1[0, :]*UtilP(Cr_tp1, 1) +
                                                ChoiceProb_tp1[1, :]*UtilP(Cw_tp1, 2)),TranIncWeights.T)
    Ev[conLen:] = numpy.squeeze(numpy.divide(-1.0, numpy.dot(numpy.expand_dims(V_T, axis=1), TranIncWeights.T)))

    # EGM step
    Cons_t[conLen:] = UtilP_inv(par.DiscFac*EutilityP[conLen:], choice)
    Coh_t[conLen:] = PdCohGrid + Cons_t[conLen:]

    # Add points to M (and C) to solve between 0 and the first point EGM finds
    # (that is add the solution in the segment where the agent is constrained)
    Coh_t[0:conLen] = numpy.linspace(0, Coh_t[conLen]*0.99, conLen)
    Cons_t[0:conLen] = Coh_t[0:conLen]

    Ev[0:conLen] = Ev[conLen+1]

    V_T = numpy.divide(-1.0, Util(Cons_t, choice) + par.DiscFac*Ev)

    # We do the envelope step in transformed value space for accuracy. The values
    # keep their monotonicity under our transformation.
    Coh_t, Cons_t, V_T = multilineEnvelope(Coh_t, Cons_t, V_T, CohGrid)

    # The solution is the working specific consumption function and value function
    # specifying lower_extrap=True for C is easier than explicitly adding a 0,
    # as it'll be linear in the constrained interval anyway.
    CFunc = LinearInterp(Coh_t, Cons_t, lower_extrap=True)
    V_TFunc = LinearInterp(Coh_t, V_T, lower_extrap=True)
    return WorkingDeatonSolution(Coh_t, Cons_t, CFunc, V_T, V_TFunc)

def rise_and_fall(x, v):
    """
    Find index vectors `rise` and `fall` such that `rise` holds the indeces `i`
    such that x[i+1]>x[i] and `fall` holds indeces `j` such that either
    x[j+1] < x[j] or x[j]>x[j-1] but v[j]<v[j-1].

    Parameters
    ----------
    x : numpy.ndarray
        array of points where `v` is evaluated
    v : numpy.ndarray
        array of values of some function of `x`

    Returns
    -------
    rise : numpy.ndarray
        see description above
    fall : numpy.ndarray
        see description above
    """
    # NOTE: assumes that the first segment is in fact increasing (forced in EGM
    # by augmentation with the constrained segment).
    # elements in common grid g

    # Identify index intervals of falling and rising regions
    # We need these to construct the upper envelope because we need to discard
    # solutions from the inverted Euler equations that do not represent optimal
    # choices (the FOCs are only necessary in these models).
    #
    # `fall` is a vector of indeces that represent the first elements in all
    # of the falling segments (the curve can potentially fold several times)
    fall = numpy.empty(0, dtype=int) # initialize with empty and then add the last point below while-loop

    rise = numpy.array([0]) # Initialize such thatthe lowest point is the first grid point
    i = 1 # Initialize
    while i <= len(x) - 2:
        # Check if the next (`ip1` stands for i plus 1) grid point is below the
        # current one, such that the line is folding back.
        ip1_falls = x[i+1] < x[i] # true if grid decreases on index increment
        i_rose = x[i] > x[i-1] # true if grid decreases on index decrement
        val_fell = v[i] < v[i-1] # true if value rises on index decrement

        if (ip1_falls and i_rose) or (val_fell and i_rose):

            # we are in a region where the endogenous grid is decreasing or
            # the value function rises by stepping back in the grid.
            fall = numpy.append(fall, i) # add the index to the vector

            # We now iterate from the current index onwards until we find point
            # where resources rises again. Unfortunately, we need to check
            # each points, as there can be multiple spells of falling endogenous
            # grids, so we cannot use bisection or some other fast algorithm.
            k = i
            while x[k+1] < x[k]:
                k = k + 1
            # k now holds either the next index the starts a new rising
            # region, or it holds the length of M, `m_len`.

            rise = numpy.append(rise, k)

            # Set the index to the point where resources again is rising
            i = k

        i = i + 1
    return rise, fall


# think! nanargmax makes everythign super ugly because numpy changed the wraning
# in all nan slices to a valueerror...it's nans, aaarghgghg
def multilineEnvelope(M, C, V_T, CohGrid):
    """
    Do the envelope step of DCEGM.

    Parameters
    ----------

    Returns
    -------


    """
    m_len = len(CohGrid)
    rise, fall = rise_and_fall(M, V_T)


    # Add the last point to the vector for convenience below
    fall = numpy.append(fall, len(M)-1)
    # The number of kinks are the number of time the grid falls
    num_kinks = len(fall)

    # Use these segments to sequentially find upper envelopes
    mV_T = numpy.empty((m_len, num_kinks))
    mV_T[:] = numpy.nan
    mC = numpy.empty((m_len, num_kinks))
    mC[:] = numpy.nan

    # understand this : # TAKE THE FIRST ONE BY HAND: prevent all the NaN-stuff..
    for j in range(num_kinks):
        # Find all common grid
        below = M[rise[j]] >= CohGrid
        above = M[fall[j]] <= CohGrid
        in_range = below + above == 0 # neither above nor below
        idxs = range(rise[j], fall[j]+1)
        m_idx_j = M[idxs]
        m_eval = CohGrid[in_range]
        mV_T[in_range,j] = LinearInterp(m_idx_j, V_T[idxs], lower_extrap=True)(m_eval)
        mC[in_range,j]  = LinearInterp(m_idx_j, C[idxs], lower_extrap=True)(m_eval) # Interpolat econsumption also. May not be nesserary

    is_all_nan = numpy.array([numpy.all(numpy.isnan(mvrow)) for mvrow in mV_T])
    # Now take the max of all these functions. Since the mV_T
    # is either NaN or very low number outside the range of the actual line-segment this works "globally"
    idx_max = numpy.zeros(CohGrid.size, dtype = int) # this one might be wrong if is_all_nan[0] == True
    idx_max[is_all_nan == False] = numpy.nanargmax(mV_T[is_all_nan == False], axis=1)

    # prefix with upper for variable that are "upper enveloped"
    upperV_T = numpy.zeros(CohGrid.size)
    upperV_T[:] = numpy.nan

    upperV_T[is_all_nan == False] = numpy.nanmax(mV_T[is_all_nan == False, :], axis=1)
    upperM = numpy.copy(CohGrid)
    # Add the zero point in the bottom
    if numpy.isnan(upperV_T[0]):
        upperV_T[0] = 0 # Since M=0 here
        mC[0]  = upperM[1]

    # Extrapolate if NaNs are introduced due to the common grid
    # going outside all the sub-line segments
    IsNaN = numpy.isnan(upperV_T)
    upperV_T[IsNaN] = LinearInterp(upperM[IsNaN == False], upperV_T[IsNaN == False], lower_extrap=True)(upperM[IsNaN])
    LastBeforeNaN = numpy.append(numpy.diff(IsNaN)>0, 0)
    LastId = LastBeforeNaN*idx_max # Find last id-number
    idx_max[IsNaN] = LastId[IsNaN]

    # Linear index used to get optimal consumption based on "id"  from max
    ncols = mC.shape[1]
    rowidx = numpy.cumsum(ncols*numpy.ones(len(CohGrid), dtype=int))-ncols
    idx_linear = numpy.unravel_index(rowidx+idx_max, mC.shape)
    upperC = mC[idx_linear]
    upperC[IsNaN] = LinearInterp(upperM[IsNaN==0], upperC[IsNaN==0])(upperM[IsNaN])

    # TODO calculate cross points of line segments to get the true vertical drops

    return upperM, upperC, upperV_T
