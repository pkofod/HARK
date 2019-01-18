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
    def __call__(self, M, V):
        # transformed_V = self.utility_inv(V)
        transformed_V = LinearInterp(M, self.utility_inv(V))
#        return lambda m: self.utility(interp(M, transformed_V, m))
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
    def __call__(self, M, N, V):
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
        transformed_V = BilinearInterp(self.utility_inv(V), M, N)
        #transformed_V = self.utility(V)

        return lambda m, n: self.utility(transformed_V(m, n))
        #return lambda m, n: self.utility_inv(interp(M, N, transformed_V, m, n))




IlliquidSaverParameters = namedtuple('IlliquidSaverParameters',
                                     'DiscFac CRRA Rliq Rilliq sigma')


class IlliquidSaverSolution(Solution):
    def __init__(self, CFuncs, BFunc, VFuncs):
        self.C = CFuncs
        self.B = BFunc
        self.VFuncs = VFuncs


class IlliquidSaver(AgentType):
    def __init__(self, DiscFac=0.98, Rliq=1.02, Rilliq=1.04,
                 CRRA=1.0, sigma=0.0,
                 TranIncVar = 0.085,
                 TranIncNodes = 6,
                 PermIncVar = 0.073,
                 PermIncNodes = 6,
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
        self.mGrid = makeGridExpMult(self.MLims[0], self.MLims[1], self.MNodes, timestonest=1)

        self.bGrid = makeGridExpMult(self.NLims[0], self.NLims[1], self.NNodes, timestonest=1)
        self.nGrid = makeGridExpMult(self.NLims[0], self.NLims[1], self.NNodes, timestonest=1)

        self.aMesh, self.bMesh = numpy.meshgrid(self.aGrid, self.bGrid, indexing = 'ij')
        self.mMesh, self.nMesh = numpy.meshgrid(self.mGrid, self.nGrid, indexing = 'ij')


        self.EGMVector = numpy.zeros(self.MNodes)

        if self.TranIncNodes == 0:
            self.TranInc = numpy.ones(1)
            self.TranIncWeights = numpy.ones(1)
        elif self.TranIncNodes >= 1:
            self.TranInc, self.TranIncWeights = numpy.polynomial.hermite.hermgauss(self.TranIncNodes)
            self.TranInc = numpy.exp(-self.TranIncVar/2.0 + sqrt(2)*sqrt(self.TranIncVar)*self.TranInc)
            self.TranIncWeights = self.TranIncWeights/sqrt(numpy.pi)
        if self.PermIncNodes == 0:
            self.PermInc = numpy.ones(1)
            self.PermIncWeights = numpy.ones(1)
        elif self.PermIncNodes >= 1:
            self.PermInc, self.PermIncWeights = numpy.polynomial.hermite.hermgauss(self.PermIncNodes)
            self.PermInc = numpy.exp(-self.PermIncVar/2.0 + sqrt(2)*sqrt(self.PermIncVar)*self.PermInc)
            self.PermIncWeights = self.PermIncWeights/sqrt(numpy.pi)

        # solve last illiquid saver
        C_nonadjust = self.mMesh
        CFunc_nonadjust = BilinearInterp(C_nonadjust, self.mGrid, self.nGrid)

        V_nonadjust = self.Util(C_nonadjust)
        VFunc_nonadjust = self.stablevalue2d(self.mGrid, self.nGrid, V_nonadjust)

        # If you adjust, transfer everything
        C_adjust = self.mMesh + (1-self.adjcost)*self.nMesh
        CFunc_adjust = BilinearInterp(C_adjust, self.mGrid, self.nGrid)

        V_adjust = self.Util(C_adjust)
        VFunc_adjust = self.stablevalue2d(self.mGrid, self.nGrid, V_adjust)

        CFuncs = (C_nonadjust, C_adjust)
        BFunc = BilinearInterp(0*self.mMesh, self.mGrid, self.nGrid)

        VFuncs = (VFunc_nonadjust, VFunc_adjust)
        self.solution_terminal = IlliquidSaverSolution(CFuncs, BFunc, VFuncs)


def solveIlliquidSaver():

    # From A, B calculate mtp1, ntp1
    for all b in mGrid:
        # We expand the dims to build a matrix
        mtp1 = Rliq*numpy.expand_dims(grid.aGrid, axis=1) + TranInc.T
        ntp1 = Rilliq*b

        P = calcDiscretePolicies(Vs)

        Ctp1 = calcCtp1((mtp1, ntp1), CFuncs, P)

        rs_augCoh = numpy.insert(rs_tp1.Coh, 0, 0.0)
        rs_augC = numpy.insert(rs_tp1.C, 0, 0.0)
        rs_augV_T = numpy.insert(rs_tp1.V_T, 0, 0.0)
        ws_augCoh = numpy.insert(ws_tp1.Coh, 0, 0.0)
        ws_augC = numpy.insert(ws_tp1.C, 0, 0.0)
        ws_augV_T = numpy.insert(ws_tp1.V_T, 0, 0.0)

        # Calculate the expected marginal utility and expected value function
        Eu[conLen:] =  par.Rfree*numpy.dot((P_tp1[0, :]*UtilP(Cr_tp1, 1) + P_tp1[1, :]*UtilP(Cw_tp1, 2)),TranIncWeights.T)
        Ev[conLen:] = numpy.squeeze(numpy.divide(-1.0, numpy.dot(numpy.expand_dims(V_T, axis=1), TranIncWeights.T)))
        # EGM step
        C[conLen:] = UtilP_inv(par.DiscFac*Eu[conLen:], choice)
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
    return None

def createCtp1(states, CFuncs, P):
    m, n = states
    Ctp1_nonadjust = CFuncs[0](m, n)
    Ctp1_adjust = CFuncs[1](m, n)

    Ctp1 = Ctp1_nonadjust*P[0] + Ctp1_adjust*P[1]
    return Ctp1
