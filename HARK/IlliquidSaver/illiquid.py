# Permanent income is not full implemented yet

import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
from HARK import Solution, AgentType
from HARK.interpolation import LinearInterp, BilinearInterp
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv, makeGridExpMult
from HARK.simulation import drawMeanOneLognormal
from math import sqrt

# === Named tuple definitions === #
Grids = namedtuple('Grids', 'm n M N a b A B EGMVector')
IlliquidParams = namedtuple('IlliquidParams', 'Rliq Rilliq DiscFac CRRA sigma adjcost')

IlliquidSaverParameters = namedtuple('IlliquidSaverParameters',
                                     'DiscFac CRRA Rliq Rilliq sigma')

Shocks = namedtuple('Shocks', 'PermInc TranInc TranIncWeights Weights')

Utility = namedtuple('Utility', 'u inv P P_inv adjcost')

class IlliquidSaverSolution(Solution):
    def __init__(self, C, CFunc, B, BFunc, V_T, V_TFunc):
        self.C = C
        self.CFunc = CFunc
        self.B = B
        self.BFunc = BFunc
        self.V_T = V_T
        self.V_TFunc = V_TFunc

class IlliquidSaver(AgentType):
    def __init__(self, DiscFac=0.98, Rliq=1.02, Rilliq=1.04,
                 CRRA=1.0, sigma=0.0,
                 TranIncVar = 0.085,
                 TranIncNodes = 6,
                 PermIncVar = 0.073,
                 PermIncNodes = 0,
                 ALims=(1e-6, 8), ANodes=900,
                 MLims=(1e-6, 15), MNodes=1000,
                 BLims=(1e-6, 8), BNodes=1000,
                 NLims=(1e-6, 15), NNodes=1000,
                 XLims=(1e-6, 10), XNodes=1000,
                 adjcost=0.02,
                 saveCommon=False,
                 **kwds):

        AgentType.__init__(self, **kwds)
        self.saveCommon = saveCommon

        self.TranIncVar = TranIncVar
        self.TranIncNodes = TranIncNodes

        self.PermIncVar = PermIncVar
        self.PermIncNodes = PermIncNodes
        self.adjcost = adjcost
        self.par = IlliquidParams(Rliq, Rilliq, DiscFac, CRRA, sigma, adjcost)

        self.time_inv = ['utility', 'grids', 'shocks', 'par']
        self.time_vary = []

        self.par = IlliquidSaverParameters(DiscFac, CRRA, Rliq, Rilliq, sigma)
        self.MNodes = MNodes
        self.ANodes = ANodes
        self.BNodes = BNodes
        self.NNodes = NNodes
        self.MLims = MLims
        self.ALims = ALims
        self.BLims = BLims
        self.NLims = NLims

        self.preSolve = self.updateLast
        self.solveOnePeriod = solveIlliquidSaver

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

        a = makeGridExpMult(self.MLims[0], self.MLims[1], self.ANodes, timestonest=1)
        m = makeGridExpMult(0, self.MLims[1], self.MNodes, timestonest=1)

        b = makeGridExpMult(self.BLims[0], self.BLims[1], self.BNodes, timestonest=1)
        n = makeGridExpMult(0, self.NLims[1], self.NNodes, timestonest=1)

        A, B = numpy.meshgrid(a, b, indexing = 'ij')
        M, N = numpy.meshgrid(m, n, indexing = 'ij')

        EGMVector = numpy.zeros(self.MNodes)

        grids = Grids(m, n, M, N, a, b, A, B, EGMVector)
        self.grids = grids

        # Create lambdas do avoid passing in parameters everywhere
        self.utility = Utility(lambda c: CRRAutility(c, self.par.CRRA),
                               lambda u: CRRAutility_inv(u, self.par.CRRA),
                               lambda c: CRRAutilityP(c, self.par.CRRA),
                               lambda u: CRRAutilityP_inv(u, self.par.CRRA),
                               self.adjcost)

        self.shocks = calcIncShks(self)

        # ### solve last illiquid saver
        # ### solve last adjuster, transfer everything
        C = grids.M + (1-self.adjcost)*grids.N
        CFunc = BilinearInterp(C, grids.m, grids.n)

        # Remember, B is the function that tells you the CHOICE of B given (m,n),
        # *not* the adjustment which would be adjustment = N-B; here N-0=N
        B = 0*grids.M
        BFunc = BilinearInterp(B, grids.m, grids.n)

        V_T = numpy.divide(-1.0, self.utility.u(C))
        V_TFunc = BilinearInterp(V_T, grids.m, grids.n)

        self.solution_terminal = IlliquidSaverSolution(C, CFunc, B, BFunc, V_T, V_TFunc)


def calcIncShks(self):
    # Construct transitory income shock nodes and weights.
    if self.TranIncNodes == 0:
        self.TranInc = numpy.ones(1)
        self.TranIncWeights = numpy.ones(1)
    elif self.TranIncNodes >= 1:
        self.TranInc, self.TranIncWeights = numpy.polynomial.hermite.hermgauss(self.TranIncNodes)
        self.TranInc = numpy.exp(-self.TranIncVar/2.0 + sqrt(2)*sqrt(self.TranIncVar)*self.TranInc)
        self.TranIncWeights = self.TranIncWeights/sqrt(numpy.pi)

    # Construct permanent income shock nodes and weights.
    if self.PermIncNodes == 0:
        self.PermInc = numpy.ones(1)
        self.PermIncWeights = numpy.ones(1)
    elif self.PermIncNodes >= 1:
        self.PermInc, self.PermIncWeights = numpy.polynomial.hermite.hermgauss(self.PermIncNodes)
        self.PermInc = numpy.exp(-self.PermIncVar/2.0 + sqrt(2)*sqrt(self.PermIncVar)*self.PermInc)
        self.PermIncWeights = self.PermIncWeights/sqrt(numpy.pi)

    self.PermInc, self.TranInc = numpy.meshgrid(self.PermInc, self.TranInc, sparse=True)
    self.PermIncWeights, self.TranIncWeights = numpy.meshgrid(self.PermIncWeights, self.TranIncWeights, sparse=True)
    self.IncWeights = self.PermIncWeights*self.TranIncWeights
    return Shocks(self.PermInc, self.TranInc, self.TranIncWeights, self.PermIncWeights*self.TranIncWeights)


def solveIlliquidSaver(solution_next, utility, grids, shocks, par):

    W = calcW(solution_next, grids, shocks, par)

    Coh_ab, CNon, CNonFunc, VNon_T, V_TNonFunc = solveIlliquidSaverConsumption(solution_next, utility, grids, shocks, par)
    BFunc = solveIlliquidSaverAdjustment(solution_next, CNonFunc, W, utility, grids, shocks, par)
    return (Coh_ab, CNon, CNonFunc, VNon_T, V_TNonFunc, BFunc)
    # BAdjustX, CAdjustX, V_TAdjustX = solveIlliquidSaverAdjustment(solution_next, CNonFunc, W, utility, grids, shocks, par)
    #
    # BAdjust = BAdjustX(grids.M+grids.N-par.adjcost)
    # CAdjust = CAdjustX(grids.M+grids.N-par.adjcost)
    #
    # V, P = calcLogSumChoiceProbs((V_TNonFunc, V_TAdjustX), par.sigma)
    #
    # B = P[0]*grids.M*0 + P[1]*BAdjust
    # BFunc = BilinearInterp(B, grids.m, grids.n)
    #
    # C = P[0]*CNon + P[1]*CAdjustX
    # CFunc = BilinearInterp(C, grids.m, grids.n)
    #
    #
    # V_T = numpy.divide(-1.0, V)
    # V_TFunc = BilinearInterp(V_T, grids.m, grids.n)
    #
    # return IlliquidSaverSolution(C, CFunc, B, BFunc, V_T, V_TFunc)


def solveIlliquidSaverConsumption(solution_next, utility, grids, shocks, par):
    Ctp1 = solution_next.CFunc # at M x N
    V_Ttp1 = solution_next.V_TFunc # at M x N

    m, n = grids.m, grids.n

    EutilityP = numpy.copy(grids.EGMVector)
    Ev = numpy.copy(grids.EGMVector)
    C = numpy.copy(grids.EGMVector)
    Coh = numpy.copy(grids.EGMVector)

    Coh_ab = numpy.zeros((len(grids.EGMVector), len(grids.b)))
    C_ab = numpy.zeros((len(grids.EGMVector), len(grids.b)))
    V_T_ab = numpy.zeros((len(grids.EGMVector), len(grids.b)))

    aLen = len(grids.a)
    conLen = len(Coh) - aLen
    ib = 0
    for b in grids.b:
        # From A, b ∈ B calculate mtp1, ntp1
        # We expand the dims to build a matrix
        mtp1 = par.Rliq*numpy.expand_dims(grids.a, axis=1) + shocks.TranInc.T
        # We're solving on b, and the return is deterministic, so ntp1 is simply
        # b + investment returns
        ntp1 = par.Rilliq*b

        Cbtp1 = Ctp1(mtp1, ntp1)
        Vb_Ttp1 =V_Ttp1(mtp1, ntp1)

        # Calculate the expected marginal utility and expected value function
        EutilityP[conLen:] = par.Rliq*utility.P(numpy.squeeze(numpy.dot(Cbtp1, shocks.TranIncWeights)))
        Ev[conLen:] = numpy.squeeze(numpy.dot(numpy.divide(-1.0, Vb_Ttp1), shocks.TranIncWeights))

        # EGM step
        C[conLen:] = utility.P_inv(par.DiscFac*EutilityP[conLen:])
        Coh[conLen:] = grids.a + C[conLen:]

        # Add points to M (and C) to solve between 0 and the first point EGM finds
        # (that is add the solution in the segment where the agent is constrained)
        Coh[0:conLen] = numpy.linspace(0, Coh[conLen]*0.99, conLen)
        C[0:conLen] = Coh[0:conLen]

        Ev[0:conLen] = Ev[conLen+1]

        V_T = numpy.divide(-1.0, utility.u(C) + par.DiscFac*Ev)
        # We do the envelope step in transformed value space for accuracy. The values
        # keep their monotonicity under our transformation.
        Coh, C, V_T = multilineEnvelope(Coh, C, V_T, grids.m)
        Coh_ab[:, ib] = Coh
        C_ab[:, ib] = C
        V_T_ab[:, ib] = V_T
        ib += 1

    CFunc = BilinearInterp(C_ab, m, n)
    V_TFunc = BilinearInterp(V_T_ab, m, n)
    return Coh_ab, C_ab, CFunc, V_T_ab, V_TFunc

def solveIlliquidSaverAdjustment(solution_next, Cstar, W, utility, grids, shocks, par):

    X = grids.m
    possibleAdj = numpy.arange(0, 0.99, 100)

    Cadjust = numpy.zeros(len(X))
    Badjust = numpy.zeros(len(X))
    Vadjust = numpy.zeros(len(X))

    cCandidates = numpy.zeros(len(X))
    bCandidates = numpy.zeros(len(X))
    vCandidates = numpy.zeros(len(X))

    xIdx = 0
    for x in X:

        candidateCounter = 0
        for adj in possibleAdj:
            b = adj*x
            m = x-b
            n = b
            c = Cstar(m, n)
            a = m - c
            bCandidates[candidateCounter] = b
            cCandidates[candidateCounter] = c
            vCandidates[candidateCounter] = utility.u(c) + par.DiscFac*W(a, b)
            candidateCounter += 1

        optimizerIdx = numpy.argmax(vCandidates)
        Cadjust[xIdx] = cCandidates[optimizerIdx]
        Badjust[xIdx] = cCandidates[optimizerIdx]
        Vadjust[xIdx] = vCandidates[optimizerIdx]
        xIdx += 1

    BFunc = LinearInterp(X, Badjust)
    CFunc = LinearInterp(X, Cadjust)
    CFunc = LinearInterp(X, Cadjust)
    VFunc = LinearInterp(X, Vadjust)

    return BFunc

def calcW(solution_next, grids, shocks, par):
    W = grids.M*0
    tiIdx = 0
    for ti in shocks.TranInc:
        mtp1 = par.Rliq*grids.M + ti
        ntp1 = par.Rilliq*grids.N
        W += numpy.divide(-1.0, solution_next.V_TFunc(mtp1, ntp1))*shocks.TranIncWeights[tiIdx]
        tiIdx += 1

    return BilinearInterp(W, grids.a, grids.b)


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
# What if there are no kinks?
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
