# Permanent income is not full implemented yet

import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
from HARK import Solution, AgentType
from HARK.interpolation import LinearInterp, BilinearInterp
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv, makeGridExpMult
from HARK.simulation import drawMeanOneLognormal
from math import sqrt
# might as well alias them utility, as we need CRRA
# should use these instead of hardcoded log (CRRA=1)
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityP_inv  = CRRAutilityP_inv
utility_inv   = CRRAutility_inv

# Transforming the value function with the inverse utility function before
# interpolating can improve precision significantly near the boundary.
class StableValue(object):
    '''
    Stable value super-class. Objects that inherent from StableValue are used to
    construct numerically stable interpolants of value functions.
    '''
    def __init__(self, utility, utility_inv):
        self.utility     = lambda c: utility(c)
        self.utility_inv = lambda u: utility_inv(u)

class StableValue1D(StableValue):
    '''
    Creates a stable value function interpolant by applying the utility and
    inverse utility before and after doing linear interpolation on the grid.
    '''
    def __init__(self, utility, utility_inv):
        StableValue.__init__(self, utility, utility_inv)
    def __call__(self, M, V, isTransformed=False):
        if isTransformed:
            transformedV = V
        else:
            transformed_V = LinearInterp(M, self.utility_inv(V))

        return lambda m: self.utility(transformed_V(m))

class StableValue2D(StableValue):
    '''
    Creates a stable value function interpolant by applying the utility and
    inverse utility before and after doing bilinear interpolation on the grid.
    An object of this class can be called with the grid vectors M, N together
    with the values V. The object call will then return a function that gives
    interpolated values of V given input (m, n).
    '''
    def __init__(self, utility, utility_inv):
        StableValue.__init__(self, utility, utility_inv)
    def __call__(self, M, N, V, isTransformed=False):
        '''
        Constructs a stable interpolation by transforming the values prior to
        evaluating the interpolant at (M, N) and applying the inverse trans-
        formation afterwards to obtain values of the original function.
        Parameters
        ----------
        M : numpy.array
            grid of values of the first state.
        N : numpy.array
            grid of values of the second state.
        V : numpy.array
            mesh of values.
        '''

        if isTransformed:
            transformedV = BilinearInterp(V, M, N)
        else:
            transformed_V = BilinearInterp(self.utility_inv(V), M, N)

        return lambda m, n: self.utility(transformed_V(m, n))

# === Named tuple definitions === #
Grids = namedtuple('Grids', 'm n M N')
IlliquidParams = namedtuple('IlliquidParams', 'Rliq Rilliq DiscFac CRRA sigma adjcost')

IlliquidSaverParameters = namedtuple('IlliquidSaverParameters',
                                     'DiscFac CRRA Rliq Rilliq sigma')

Shocks = namedtuple('Shocks', 'PermInc TranInc Weights')

Utility = namedtuple('Utility', 'u inv P P_inv lambda')

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
                 LiqNodes=2000,
                 MLims=(1e-6, 10), MNodes=100,
                 NLims=(1e-6, 10), NNodes=100,
                 XLims=(1e-6, 10), XNodes=100,
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

        self.time_inv = ['AGrid', 'LiqGrid','BGrid', 'IlliqGrid', 'EGMVector',
                         'par', 'Util', 'UtilP', 'UtilP_inv', 'saveCommon', 'TranInc', 'TranIncWeights']
        self.time_vary = ['age']

        self.par = IlliquidSaverParameters(DiscFac, CRRA, Rliq, Rilliq, sigma)
        self.MLims = MLims
        self.MNodes = MNodes
        self.NLims = NLims
        self.NNodes = NNodes

        # - 10.0 moves curve down to improve lienar interpolation
        self.Util = lambda c: utility(c, CRRA) - 10.0
        self.Util_inv = lambda u: utility_inv(u + 10.0, CRRA) # ... so ...
        self.UtilP = lambda c: utilityP(c, CRRA) # we require CRRA 1.0 for now...
        self.UtilP_inv = lambda u: utilityP_inv(u, CRRA) # ... so ...

        self.stablevalue   = StableValue1D(self.Util, self.Util_inv)
        self.stablevalue2d = StableValue2D(self.Util, self.Util_inv)

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

        self.aGrid = makeGridExpMult(self.MLims[0], self.MLims[1], self.MNodes, timestonest=1)
        m = makeGridExpMult(0, self.MLims[1], self.MNodes, timestonest=1)

        self.bGrid = makeGridExpMult(self.NLims[0], self.NLims[1], self.NNodes, timestonest=1)
        n = makeGridExpMult(0, self.NLims[1], self.NNodes, timestonest=1)

        self.aMesh, self.bMesh = numpy.meshgrid(self.aGrid, self.bGrid, indexing = 'ij')
        M, N = numpy.meshgrid(m, n, indexing = 'ij')

        grids = Grids(m, n, M, N)
        self.grids = grids

        self.EGMVector = numpy.zeros(self.MNodes)

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
        self.shocks = Shocks(self.PermInc, self.TranInc, self.PermIncWeights*self.TranIncWeights)

        # ### solve last illiquid saver
        # ### solve last adjuster, transfer everything
        C = grids.M + (1-self.adjcost)*grids.N
        CFunc = BilinearInterp(C, grids.m, grids.n)

        # Remember, B is the function that tells you the CHOICE of B given (m,n),
        # *not* the adjustment which would be adjustment = N-B; here N-0=N
        B = 0*grids.M
        BFunc = BilinearInterp(B, grids.m, grids.n)

        V_T = numpy.divide(-1.0, self.Util(C))
        V_TFunc = BilinearInterp(V_T, grids.m, grids.n)

        self.solution_terminal = IlliquidSaverSolution(C, CFunc, B, BFunc, V_T, V_TFunc)


def solveIlliquidSaver(solution_next, EGMVector, grids, shocks, par):
    consumptionSolution = solveIlliquidSaverConsumption(solution_next, EGMVector, grids, shocks, par)
    adjustmentSolution = solveIlliquidSaverAdjustment(solution_next, consumptionSolution, grids, shocks, par)

    return IlliquidSaverSolution(C, CFunc, B, BFunc, V_T, V_TFunc)

def solveIlliquidSaverConsumption(solution_next, EGMVector, grids, shocks, par):
    Ctp1 = solution_next.CFunc # at M x N
    V_Ttp1 = solution_next.V_TFunc # at M x N

    aLen = len(grids.a)
    conLen = len(grids.m) - aLen

    EutilityP = numpy.copy(EGMVector)
    Ev = numpy.copy(EGMVector)

    for b in grids.n:
        # From A, b ∈ B calculate mtp1, ntp1
        # We expand the dims to build a matrix
        mtp1 = par.Rliq*numpy.expand_dims(grids.a, axis=1) + shocks.TranInc.T
        # We're solving on b, and the return is deterministic, so ntp1 is simply
        # b + investment returns
        ntp1 = par.Rilliq*b

        Cbtp1 = Ctp1(mtp1, ntp1)
        Vb_Ttp1 =V_Ttp1(mtp1, ntp1)

        # Calculate the expected marginal utility and expected value function
        EutilityP[conLen:] = par.Rfree*numpy.dot(Cbtp1, shocks.TranIncWeights.T)
        Ev[conLen:] = numpy.squeeze(numpy.divide(-1.0, numpy.dot(numpy.expand_dims(Vb_Ttp1, axis=1), shocks.TranIncWeights.T)))

        # EGM step
        C[conLen:] = UtilP_inv(par.DiscFac*EutilityP[conLen:])
        Coh[conLen:] = PdCohGrid + C[conLen:]

        # Add points to M (and C) to solve between 0 and the first point EGM finds
        # (that is add the solution in the segment where the agent is constrained)
        Coh[0:conLen] = numpy.linspace(numpy.min(PdCohGrid), Coh[conLen]*0.99, conLen)
        C[0:conLen] = Coh[0:conLen]

        Ev[0:conLen] = Ev[conLen+1]

        V_T = numpy.divide(-1.0, Util(C, choice) + par.DiscFac*Ev)
        # We do the envelope step in transformed value space for accuracy. The values
        # keep their monotonicity under our transformation.
        Coh, C, V_T = multilineEnvelope(Coh, C, V_T, CohGrid)

        CFunc
        return IlliquidSaverSolution(C, CFunc, B, BFunc, V_T, V_TFunc)

def solveIlliquidSaverAdjustment(solution_next, consumptionSolution, W, grids, shocks, par):
    Cstar = consumptionSolution.CFunc

    X = grids.m
    possibleAdj = [0.0, 0.25, 0.5, 0.75, 0.99]

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
            vCandidates[candidateCounter] = Util(c) + par.DiscFac*W(a, b)
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
