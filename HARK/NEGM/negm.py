
import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
from HARK import Solution, AgentType
from HARK.interpolation import LinearInterp
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv
from HARK.simulation import drawMeanOneLognormal
from math import sqrt
# might as well alias them utility, as we need CRRA
# should use these instead of hardcoded log (CRRA=1)
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityP_inv  = CRRAutilityP_inv
utility_inv   = CRRAutility_inv


IlliquidSaverParameters = namedtuple('IlliquidSaverParameters',
                                     'DiscFac CRRA DisUtil Rliq Rilliq YRet YWork sigma')


class IlliquidSaverSolution(Solution):
    def __init__(self, C, B, Vs, P):
        self.C = C
        self.B = B
        self.Vs = Vs
        self.P = P


class RetiringDeaton(AgentType):
    def __init__(self, DiscFac=0.98, Rliq=1.02, Rilliq=1.04,
                 CRRA=1.0, sigma=0.0,
                 TranIncVar = 0.005,
                 TranIncNodes = 0,
                 CohNodes=2000,
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

        self.par = IlliquidSaverParameters(DiscFac, CRRA, DisUtil, Rliq, Rilliq YRet, YWork, sigma)
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
        self.CohGrid = nonlinspace(self.PdCohLims[0], self.PdCohLims[1]*1.5, self.PdCohNodes)
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
            Crs = LinearInterp(rs.Coh, rs.C, lower_extrap=True)(commonM)
            Cws = ws.C
    #        Cws = LinearInterp(rs.Coh, rs.C)(CohGrid)

            V_Trs = LinearInterp(rs.Coh, rs.V_T, lower_extrap=True)(commonM)
            V_Tws = ws.V_T

            # use vstack to stack the choice specific value functions by
            # value of cash-on-hand
            V_T, P = discreteEnvelope(numpy.stack((V_Trs, V_Tws)), self.par.sigma)

            # find the expected consumption prior to discrete choice by element-
            # wise multiplication between choice distribution and choice specific
            # consumption functions
            C = (P*numpy.stack((Crs, Cws))).sum(axis=0)
        else:
            C, V_T, P = None, None, None

        self.solution_terminal = RetiringDeatonSolution(rs, ws, commonM, C, V_T, P)#M, C, -1.0/V, P)
