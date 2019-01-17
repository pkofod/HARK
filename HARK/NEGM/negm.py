
import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
from HARK import Solution, AgentType
from HARK.interpolation import LinearInterp
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv, makeGridExpMult
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
                 LiqNodes=2000,
                 PdLiqLims=(1e-6, 700), PdLiqNodes=1800,
                 saveCommon=False,
                 **kwds):

        AgentType.__init__(self, **kwds)
        self.saveCommon = saveCommon

        self.TranIncVar = TranIncVar
        self.TranIncNodes = TranIncNodes

        self.time_inv = ['PdLiqGrid', 'LiqGrid','PdIlliqGrid', 'IlliqGrid', 'EGMVector',
                         'par', 'Util', 'UtilP', 'UtilP_inv', 'saveCommon', 'TranInc', 'TranIncWeights']
        self.time_vary = ['age']

        self.par = IlliquidSaverParameters(DiscFac, CRRA, Rliq, Rilliq, sigma)
        self.PdLiqLims = PdLiqLims
        self.PdLiqNodes = PdLiqNodes
        self.LiqNodes = LiqNodes

        # - 10.0 moves curve down to improve lienar interpolation
        self.Util = lambda c, d: utility(c, CRRA) - self.par.DisUtil*(2-d) - 10.0
        self.UtilP = lambda c, d: utilityP(c, CRRA) # we require CRRA 1.0 for now...
        self.UtilP_inv = lambda u, d: utilityP_inv(u, CRRA) # ... so ...

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

        self.PdLiqGrid = makeGridExpMult(self.PdLiqLims[0],  self.PdLiqLims[1], self.PdLiqNodes, timestonest=1)
        self.LiqGrid = makeGridExpMult(self.PdLiqLims[0], self.PdLiqLims[1]*1.5, self.PdLiqNodes, timestonest=1)

        self.PdIlliqGrid = makeGridExpMult(self.PdLiqLims[0],  self.PdLiqLims[1], self.PdLiqNodes, timestonest=1)
        self.IlliqGrid = makeGridExpMult(self.PdLiqLims[0], self.PdLiqLims[1]*1.5, self.PdLiqNodes, timestonest=1)

        self.EGMVector = numpy.zeros(self.LiqNodes)

        if self.TranIncNodes == 0:
            self.TranInc = numpy.ones(1)
            self.TranIncWeights = numpy.ones(1)
        elif self.TranIncNodes >= 1:
            self.TranInc, self.TranIncWeights = numpy.polynomial.hermite.hermgauss(self.TranIncNodes)
            self.TranInc = numpy.exp(-self.TranIncVar/2.0 + sqrt(2)*sqrt(self.TranIncVar)*self.TranInc)
            self.TranIncWeights = self.TranIncWeights/sqrt(numpy.pi)

        # solve last illiquid saver
        C = 1
        B = 1
        Vs = 1
        P = 1
        self.solution_terminal = IlliquidSaverSolution(C, B, Vs, P)

    def solveLastIlliquidSaver(self):
        """
        Solves the last period of an agent that saves in both a liquid and an
        illiquid asset.

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
        # this transformation is different than in our G2EGM, we
        # need to figure out which is better

        CFunc = lambda coh: LinearInterp(Coh, C, lower_extrap=True)(coh)
        V_T = -1.0/self.Util(self.CohGrid, choice)
        VFunc = lambda coh: LinearInterp(Coh, V_T, lower_extrap=True)(coh)
        return RetiredDeatonSolution(Coh, C, CFunc, V_T, VFunc)
