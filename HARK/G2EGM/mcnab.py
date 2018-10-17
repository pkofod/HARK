from collections import namedtuple
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv
from HARK.interpolation import LinearInterp, BilinearInterp
from HARK import Solution, AgentType
#from g2egm import segmentUpperEnvelope, discreteEnvelope, StableValue1D, StableValue2D, nonlinspace, w_and_wP, cleanSegment
import numpy
from numba import njit
import math
#################################
# - stable value interpolants - #
#################################

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

def w_and_wP(rs, ws, grids, par, discreteEnvelope, stablevalue2d):
    '''
    Construct undiscounted post-decision value function and derivatives wrt the
    two post decision states.

    Parameters
    ----------
    rs : RetirementSolution
    ws : WorkingSolution
    grids : Grids
        a named tuple with grids and meshes
    par : MCNABParameters
        an object with utility parameters
    discreteEnvelope : function
        function to calculate the discrete choices given choice specific value
        functions and the associated over all value
    stablevalue2d : StableValue2D
        a function to construct an interpolant that is stable in some sense (see
        docstring for StableValue2D for more precise information)
    '''

    # In this function a and b values are understood to be at period t just before
    # transitioning into t+1. Similarly, (m,n) grids and meshes are just at the
    # beginning of t+1 where the agent has to decide what to do (so before the)
    # discrete and continuous choies are made.

    # =============== #
    # -  meshgrids  - #
    # =============== #

    # The post decision grids
    aInterp, bInterp = grids.aGridPost, grids.bGridPost

    # m as a plus the return; before income
    mInterp = aInterp*par.Ra + par.eta
    # n as b plus the return
    nInterp = bInterp*par.Rb
    # create (m, n) meshes of
    MInterp, NInterp = numpy.meshgrid(mInterp, nInterp, indexing='ij')

    # ===================================== #
    # - values given discrete choice      - #
    # ===================================== #
    # retirement value; includes income
    # rv = rs.v(M.flatten(), N.flatten()).reshape(M.shape)
    vInterpRet = rs.v(MInterp, NInterp)
    # working value   ; includes income
    # wv = ws.v(m_next, n_next)
    vInterpWork = ws.v(MInterp, NInterp)

    # ===================================== #
    # - discrete envelope at interp grids - #
    # ===================================== #
    wInterp, Pr_1 = discreteEnvelope(numpy.array([vInterpRet, vInterpWork]))

    # ===================================== #
    # - post decision value function      - #
    # ===================================== #
    w = stablevalue2d(aInterp, bInterp, wInterp)

    # ==================================================== #
    # - post decision value function partial derivatives - #
    # ==================================================== #

    # pre-calculate Pr(z=0) = 1 - Pr(z=1) mostly for clarity
    # as llvm should be able to optimize this away on its own
    Pr_0 = (1 - Pr_1)
    wPaInterp = par.Ra*(rs.vPm(MInterp, NInterp)*Pr_0 +
                        ws.vPm(mInterp, nInterp)*Pr_1)
    wPbInterp = par.Rb*(rs.vPn(MInterp, NInterp)*Pr_0 +
                        ws.vPn(mInterp, nInterp)*Pr_1)

    # Construct interpolants for wPa and wPb
    wPa = stablevalue2d(aInterp, bInterp, wPaInterp)
    wPb = stablevalue2d(aInterp, bInterp, wPbInterp)

    return w, wPa, wPb

# =========================== #
# - barycentric interpolant - #
# =========================== #

def discreteEnvelope(Vs, sigma):
    '''
    Returns the final optimal value and policies given the choice specific value
    functions Vs. Policies are degenerate if sigma == 0.0.

    Parameters
    ----------
    Vs : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.

    sigma : float
        A number that controls the variance of the taste shocks

    Returns
    -------
    V : [numpy.array]
        A numpy.array that holds the integrated value function.
    P : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    '''
    if sigma == 0.0:
        # Is there a way to fuse these?
        P = numpy.argmax(Vs, axis=0)
        V = numpy.amax(Vs, axis=0)
        return V, P

    maxV = Vs.max()
    P = numpy.exp(Vs/sigma)/numpy.sum(numpy.exp(Vs/sigma))
    V = maxV + sigma*numpy.log(numpy.sum(numpy.exp((Vs-maxV)/sigma)))
    return V, P


@njit(error_model="numpy")
def segmentUpperEnvelope(grids, cG, dG, vG, mSegment, nSegment, cSegment, dSegment, vSegment, best):
    """
    segmentUpperEnvelope calculates the upper envelope to a segment while
    re-interpolating it down to a common grid spanned by mGrid and nGrid.
    """
    # the approach here is to construct a moving window made
    # of four points i0, i1, i2, i3 that together makes up
    # two simpleces forming a square (since it's 2D)

    mGrid, nGrid = grids.mGrid, grids.nGrid
    nm, nn = mSegment.shape

    # These are the policies and values at the verteces of the current simplex. C is
    # the first control D is the second and V is the value.
    C, D, V = numpy.zeros(3), numpy.zeros(3), numpy.zeros(3)

    # it is the current simplex in matrix form (each row is a vertex)
    simplex = numpy.zeros((3, 2))

    # Loop over all the segment m and n's. We loop for i_m's in [0, nm-1] because we
    # add one to the i_m indeces when building the simpleces. Likewise for nn
    for i_m in range(nm-1):     # subtract one to stay in bounds...
        for i_n in range(nn-1): # ... because we add one below
            # we loop over i_s [0,1] to build two simpleces that make up a hypercube
            for i_s in range(2):
                # define verteces
                i0 = (i_m + i_s, i_n + i_s) # either i_s is either 0 or 1
                i1 = (i_m + 1, i_n)
                i2 = (i_m, i_n + 1)
                m0, n0 = mSegment[i0], nSegment[i0]
                m1, n1 = mSegment[i1], nSegment[i1]
                m2, n2 = mSegment[i2], nSegment[i2]

                # construct the simplex, each row is a vertex
                simplex[0] = m0, n0
                simplex[1] = m1, n1
                simplex[2] = m2, n2

                m_min, n_min = min(m0, m1, m2), min(n0, n1, n2)
                m_max, n_max = max(m0, m1, m2), max(n0, n1, n2)

                ms = numpy.searchsorted(mGrid, numpy.array([m_min, m_max]))
                ns = numpy.searchsorted(nGrid, numpy.array([n_min, n_max]))

                len_m, len_n = len(mGrid), len(nGrid)

                # inverse of denominator for barycentric weights
                denominator = ((n1 - n2)*(m0 - m2) + (m2 - m1)*(n0 - n2))
                inv_denominator = 1.0/denominator

                d12 = n1 - n2
                d21 = m2 - m1
                d20 = n2 - n0
                d02 = m0 - m2

                C[:] = cSegment[i0], cSegment[i1], cSegment[i2]
                D[:] = dSegment[i0], dSegment[i1], dSegment[i2]
                V[:] = vSegment[i0], vSegment[i1], vSegment[i2]


                for i_mg in range(ms[0], min(ms[1]+1, len_m)):
                    m_i = mGrid[i_mg]
                    for i_ng in range(ns[0], min(ns[1]+1, len_n)):
                        # (m, n) is the common grid point in R^2
                        n_i = nGrid[i_ng]

                        weight_A = (d12*(m_i - m2) + d21*(n_i - n2))*inv_denominator
                        weight_B = (d20*(m_i - m2) + d02*(n_i - n2))*inv_denominator

                        # Is an array the best thing to do here? It makes
                        # JITting easy, but it seems wasteful
                        weights = numpy.array([weight_A, weight_B, 1.0 - weight_A - weight_B])

                        updateSegment(i_mg, i_ng, weights, C, D, V, cG, dG, vG, best)

# @njit # something is not numba
# can't use boolean slicesin numba
def cleanSegment(T, V):
    """
    cleanSegment uses the elements of T to create indeces of invalid entries in
    t in T and in V. Used to rule out infeasible policies.

    Parameters
    ----------
    T : tuple of numpy.array
        tuple of arrays used to invalidate
    V : numpy.array
        the value associated

    Returns
    -------
    Nothing
    """

    threshold = 0.0
    isinfeasible = T[0] < threshold
    for i in range(1, len(T)):
        isinfeasible = numpy.logical_or(isinfeasible, (T[i] < threshold))

    for t in T:
        t[isinfeasible] = numpy.nan

    V[isinfeasible] = numpy.nan

@njit
def updateSegment(i_mg, i_ng, Weights, C, D, V, CG, DG, VG, best):
    """
    vBest is currently best at x (wehere weights are calculated)
    """
    # if weights are within the cutoff, go ahead and interpolate, and update

    Wsum = Weights.sum()
    tol = 0.1
    if (Wsum < (1.0 + tol)) & (Wsum > (1.0 - tol)):
#    if numpy.any(Weights < -0.25):
        # interpolate
        c = numpy.dot(Weights, C)
        d = numpy.dot(Weights, D)
        v = numpy.dot(Weights, V)

        # check if v is better than currently best recorded v at this (m, n)
        # and if consumption and deposits have sensible values.
        sensible_policies = (d >= 0.0) & (c >= 0.0)

        if (v > VG[i_mg, i_ng]) & sensible_policies:
            CG[i_mg, i_ng] = c
            DG[i_mg, i_ng] = d
            VG[i_mg, i_ng] = v
            best[i_mg, i_ng] = True



def g(d, chi = 0.1):
    '''
    Returns the additional increase in illiquid assets given the amount deposited.

    Parameters
    ----------
    d : float
        amount deposited into illiquid assets account
    chi : float
        parameter that partially determines incentive to deposit. Defaults to 0.1.

    Returns
    -------
    gd : float
        the increase in illiquid assets above and beyond d itself
    '''
    return chi*numpy.log(1 + d)

# A named tuple to holds grids - simply for easier passing around
Grids = namedtuple('Grids', 'aGrid bGrid aGrid_ret aGridPost bGridPost mGrid nGrid mGrid_ret aMesh bMesh mMesh nMesh mGrid_con nGrid_con mMesh_con nMesh_con')

class RetirementSolution(Solution):
    """
    RetirementSolution holds the interpolants related to the retirement specific
    solution in a given period.
    """
    def __init__(self, c1d, c, v1d, v, vPm, vPn):
        """
        Construct an instance of RetirementSolution.
        """
        self.c1d = c1d
        self.c = c
        # Savings into pension fund is 0 when retired
        self.d1d = lambda m: numpy.zeros(m.shape)
        self.d = lambda m, n: numpy.zeros(m.shape)
        self.v1d = v1d
        self.v = v
        self.vPm = vPm
        self.vPn = vPn


class WorkingSolution(Solution):
    def __init__(self, aMesh, bMesh,
                 mMesh, nMesh,
                 cMesh, dMesh, vMesh,
                 vPmMesh, vPnMesh,
                 c, d, v, vPm, vPn, ucon, con, dcon, acon):
        self.aMesh = aMesh
        self.bMesh = bMesh
        self.mMesh = mMesh
        self.nMesh = nMesh
        self.cMesh = cMesh
        self.dMesh = dMesh
        self.vMesh = vMesh
        self.vPmMesh = vPmMesh
        self.vPnMesh = vPnMesh
        self.c = c
        self.d = d
        self.v = v
        self.vPm = vPm
        self.vPn = vPn
        self.ucon = ucon
        self.con = con
        self.dcon = dcon
        self.acon = acon

class MCNABSolution(Solution):
    def __init__(self, rs, ws, Pr_1, vMesh, v, w, wPa, wPb):
        self.rs = rs
        self.ws = ws
        self.Pr_1 = Pr_1
        self.vMesh = vMesh
        self.v = v
        self.w = w
        self.wPa = wPa
        self.wPb = wPb

class MCNABParameters(object):
    def __init__(self, Ra, Rb, CRRA, chi, alpha, DiscFac, y, eta, sigma):
        self.Ra = Ra
        self.Rb = Rb
        self.CRRA = CRRA
        self.chi = chi
        self.alpha = alpha
        self.DiscFac = DiscFac
        self.y = y
        self.eta = eta
        self.sigma = sigma

# class Utility(object):
#     def __init__(self, Ra, Rb, CRRA, chi, alpha, DiscFac, y, eta, sigma):
#         self.Ra = Ra
#         self.Rb = Rb
#         self.CRRA = CRRA
#         self.chi = chi
#         self.alpha = alpha
#         self.DiscFac = DiscFac
#         self.y = y
#         self.eta = eta
#         self.sigma = sigma
#


Utility = namedtuple('Utility', 'u inv P P_inv g')


class MCNAB(AgentType):
    def __init__(self, T = 20,
                 Ra = 1.02, Rb = 1.04, # interest rates
                 m_min = 1e-6, n_min = 1e-6,
                 m_max = 10.0, n_max = 12.0,
                 m_max_ret = 70,

                 a_min = 1e-6, b_min = 1e-6,
                 a_max = 18.0, b_max = 18.0,

                 a_min_ret = 1e-6, a_max_ret = 50.0,

                 Na = 250, Nb = 250,
                 Na_w = 250, Nb_w = 250,

                 Nm = 500, Nn = 500,

                 Nm_ret = 500, Na_ret = 500,
                 y = 0.5, eta = 1.0,
                 CRRA = 2.0, alpha = 0.25, chi = 0.1,
                 DiscFac = 0.98,
                 sigma = 0.0, verbose = True,
                 **kwds):

        # Initialize a basic AgentType
        AgentType.__init__(self, **kwds)

        self.time_inv = ['par', 'grids', 'utility', 'discreteEnvelope',
                         'stablevalue',  'stablevalue2d', 'verbose']
        self.time_vary = ['age']

        self.age = list(range(T-1))
        self.solveOnePeriod = solve_period

        # Terminal period is T
        self.T = T

        # Common grid
        self.m_max = m_max
        self.n_max = n_max
        self.Nm = Nm
        self.Nn = Nn

        mGrid = nonlinspace(m_min, m_max, Nm)
        mGrid_ret = nonlinspace(m_min, m_max_ret, Nm_ret)
        nGrid = nonlinspace(n_min, n_max, Nn)
        mAdd_con = 3
        nAdd_con = 3
        mGrid_con = nonlinspace(m_min, m_max+mAdd_con, Nm)
        nGrid_con = nonlinspace(n_min, n_max+nAdd_con, Nn)

        # (a, b) grids
        aGrid = nonlinspace(a_min, a_max, Na)
        bGrid = nonlinspace(b_min, b_max, Nb)
        aGrid_ret = nonlinspace(a_min_ret, a_max_ret, Na_ret)

        aMesh,     bMesh     = numpy.meshgrid(aGrid, bGrid, indexing = 'ij')

        # Store a few mesh grids for future use
        mMesh, nMesh = numpy.meshgrid(mGrid, nGrid, indexing = 'ij')
        mMesh_con, nMesh_con = numpy.meshgrid(mGrid_con, nGrid_con, indexing = 'ij')

        aGridPost = nonlinspace(a_min, a_max, Na_w)
        bGridPost = nonlinspace(b_min, b_max, Nb_w)

        # pack grids
        self.grids = Grids(aGrid, bGrid,
                           aGrid_ret,
                           aGridPost, bGridPost,
                           mGrid, nGrid, mGrid_ret,
                           aMesh, bMesh,
                           mMesh, nMesh,
                           mGrid_con, nGrid_con,
                           mMesh_con, nMesh_con)

        # ============== #
        # - parameters - #
        #=============== #

        # Ra is returns to liquid asset, Rb is returns to illiquid asset,
        # CRRA is the CRRA parameter in the utility function , chi is the
        # deposit parameter, alpha is disutility of work, and DiscFac is the
        # discount factor, eta is the income when working, and sigma is the
        # discrete choice smoother
        self.par = MCNABParameters(Ra, Rb, CRRA,
                                   chi, alpha, DiscFac,
                                   y, eta, sigma)

        # The .updateLast method is stored in .preSolve such
        # that the terminal solution does not get out of sync
        # with any model parameters that might change between
        # calls to .solve . NOTE: grids are
        # currently not updated, so this actually doesn't fully
        # work. Utility parameters can be updated though.
        self.preSolve = self.updateLast

        # set verbosity to True by default
        self.verbose = verbose

    def discreteEnvelope(self, Vs):
        '''
        Calculate the discrete upper envelope given model parameters.

        Parameters
        ----------
        Vs : numpy.array
            array of choice specific values stored in last dimension

        Returns
        -------
        (V, P) : (numpy.array, numpy.array)
            tuple of integrated value function and policy matrix
        '''
        return discreteEnvelope(Vs, self.par.sigma)

    def updateLast(self):
        '''
        Solve last period of problem. This method is actually called by preSolve,
        to make sure that the terminal period is updated if parameters are changed
        and the model is resolved. To this end, it also updates the lambdas used
        in the Utility instance.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        par = self.par
        # Create lambdas do avoid passing in parameters everywhere
        self.utility = Utility(lambda c: CRRAutility(c, par.CRRA),
                               lambda u: CRRAutility_inv(u, par.CRRA),
                               lambda c: CRRAutilityP(c, par.CRRA),
                               lambda u: CRRAutilityP_inv(u, par.CRRA),
                               lambda d: g(d, par.chi))

        # Stable value function interpolants. They apply the utility function beforew
        # interpolating and apply the inverse utility functino after interpolating
        # for increase accuracy - especially in the lower values of the states.
        self.stablevalue   = StableValue1D(self.utility.u, self.utility.inv)
        self.stablevalue2d = StableValue2D(self.utility.u, self.utility.inv)

        # Solve the retirement sub-problem (orthogonal to the working problem)
        rs = self.updateLastRetirement()
        # Solve the working sub-problem (orthogonal to the retirement problem)
        ws = self.updateLastWorking()

        #### Find optimal choices and values
        # Note that discrete choices will currently be degenerate as we
        # don't have a taste shock implemented as a choice in the AgentType.

        # Unpack post-decision state grids
        aGridPost = self.grids.aGridPost
        bGridPost = self.grids.bGridPost

        # We need an upper envelope of the two values prior to the discrete choice
        m, n = self.grids.mGrid, self.grids.nGrid
        M, N = self.grids.mMesh, self.grids.nMesh

        # retirement y for this period is not paid until end of the life and
        # there is no bequest motive
        rv_envelope = rs.v(M, N)
        # labor income for this y is not paid until end of the life and there
        # is no bequest motive
        wv_envelope = ws.v(M, N)
        vs = numpy.array([rv_envelope, wv_envelope]) # fixme preallocate

        vMesh, Pr_1 = self.discreteEnvelope(vs)
        v = self.stablevalue2d(m, n, vMesh)

        w, wPa, wPb = w_and_wP(rs, ws, self.grids, self.par, self.discreteEnvelope, self.stablevalue2d)

        # Store solution in solution_terminal
        self.solution_terminal = MCNABSolution(rs, ws, # solution objects
                                               Pr_1,   # Pr(z=1) on (M,N)
                                               vMesh, v,
                                               w, wPa, wPb) # value at previous period's PD states

    def updateLastRetirement(self):
        """
        Updates the last period's retirement sub-problem solution

        Parameters
        ---------
        None

        Returns
        -------
        rs : RetirementSolution
            holds variables relevant to the retirement solution
        """
        # Method:
        #     1) Simply consume everthing in ultimate period
        #        -> set consumption to total ressources
        #     2) Construct bi-linear interpolant on (m,n)-space
        #     3) Construct value as value of cstar as this is last period
        #     4) Construct value derivatives as utility derivative

        # Grab grids from grid-holder
        mGrid = self.grids.mGrid

        CGrid = self.grids.mGrid

        c1DFunc = LinearInterp(mGrid, CGrid)
        cFunc = lambda m, n: c1DFunc(m + n)

        # Create interpolant for choice specific value function at T
        vMesh = self.utility.u(CGrid)
        v1DFunc = self.stablevalue(mGrid, vMesh)
        vFunc = lambda m, n: v1DFunc(m+n)

        # In the retired state and in last period, the derivative
        # of the value function is simply the marginal utility.
        vPm1DFunc = self.stablevalue(mGrid, self.utility.P(CGrid))
        vPmFunc = lambda m, n: vPm1DFunc(m+n)
        vPnFunc = vPmFunc

        # Collect retirement relevant solution
        rs = RetirementSolution(c1DFunc, cFunc, v1DFunc, vFunc, vPmFunc, vPnFunc)
        return rs

    def updateLastWorking(self):
        # Method:
        #     1) Simply consume everthing in ultimate period
        #        -> set consumption to total ressources
        #     2) Construct bi-linear interpolant on (m,n)-space
        #     3) Construct value as value of cstar as this is last period
        #     4) Construct value derivatives as utility derivative

        # Grab grids from grid-holder
        mGrid = self.grids.mGrid
        nGrid = self.grids.nGrid

        cMesh = self.grids.mMesh + self.grids.nMesh
        dMesh = numpy.zeros(cMesh.shape)

        c = self.stablevalue2d(mGrid, nGrid, cMesh)
#        c = RectBivariateSpline(cMesh, mGrid, nGrid)
        d = lambda m, n: numpy.zeros(m.shape)

        # Create interpolant for choice specific

        vMesh = self.utility.u(cMesh) - self.par.alpha

        v = self.stablevalue2d(mGrid, nGrid, vMesh)

        vPmMesh = self.utility.P(cMesh)
        vPnMesh = numpy.copy(vPmMesh)

        vPm = self.stablevalue2d(mGrid, nGrid, vPmMesh)
        vPn = self.stablevalue2d(mGrid, nGrid, vPnMesh)

        return WorkingSolution(0, 0, self.grids.mMesh, self.grids.nMesh, cMesh, dMesh, vMesh, vPmMesh, vPnMesh, c, d, v, vPm, vPn, (1), (1), (1), (1))


def solve_period(solution_next, par, grids, utility, discreteEnvelope,  stablevalue,  stablevalue2d, verbose):
    # =============================
    # - unpack from solution_next -
    # =============================
    rs_tp1 = solution_next.rs # next period retired solution

    # grids
    mGrid, nGrid = grids.mGrid, grids.nGrid

    # ===========
    # - retired -
    # ===========
    if verbose:
        print("Solving retired persons sub-problem")
    rs = solve_period_retirement(rs_tp1, par, grids, utility, stablevalue)

    # ===========
    # - working -
    # ===========
    if verbose:
        print("Solving working persons sub-problem")
    ws = solve_period_working(solution_next, par, grids, utility, stablevalue, stablevalue2d, verbose)

    # =====================================
    # - construct discrete upper envelope -
    # =====================================
    M = grids.mMesh
    N = grids.nMesh
    # rs_v_interp = rs.v(M.flatten(), N.flatten()).reshape(M.shape)
    # ws_v_interp = ws.v(mGrid, nGrid)
    vInterpRet = rs.v(M, N)
    vInterpWork = ws.v(M, N)

    vInterp, Pr_1 = discreteEnvelope(numpy.array([vInterpRet, vInterpWork]))

    v = stablevalue2d(mGrid, nGrid, vInterp)

    # ==========================
    # - make w and derivatives -
    # ==========================
    w, wPa, wPb = w_and_wP(rs, ws, grids, par, discreteEnvelope, stablevalue2d)

    return MCNABSolution(rs, ws,
                         Pr_1, vInterp, v,
                         w, wPa, wPb)


def solve_period_working(solution_next, par, grids, utility, stablevalue, stablevalue2d, verbose):
    """
    Solves a period of the retirement model using EGM.

    Parameters
    ----------
    rs_tp1 : RetirementSolution
        holds the retirement solution for the next period
    par : MCNABParameters
        object that holds parameters relevant to utility and transitions
    grids : Grids
        holds all the grids and meshes for (m, n) and (a, b) space
    utility : Utility
        holds utility functions, inverses, derivatives, ...
    stablevalue : StableValue1D
        is used to create a stable interpolant of value functions

    Returns
    -------
    rs : RetirementSolution
        holds the retirement solution for the current period
    """

    if verbose:
        print("... solving working problem")

    # =================================
    # - next period w and derivatives -
    #==================================
    w_tp1 = solution_next.w
    wPa_tp1 = solution_next.wPa
    wPb_tp1 = solution_next.wPb

    # ====================
    # - grids and meshes -
    # ====================
    mGrid, nGrid = grids.mGrid, grids.nGrid
    mMesh, nMesh = grids.mMesh, grids.nMesh

    aMesh, bMesh = grids.aMesh, grids.bMesh

    # shape of common grid
    grid_shape = (len(mGrid), len(nGrid))

    # solution arrays for ...
    # ... consumption levels
    cMesh = numpy.zeros(grid_shape)
    # ... deposit levels
    dMesh = numpy.zeros(grid_shape)
    # ... value function
    vMesh = -numpy.ones(grid_shape)*numpy.inf

    # ================
    # - wP's on mesh -
    # ================
    # wPaMesh = wPa(aGrid, bGrid)
    # wPbMesh = wPb(aGrid, bGrid)
    wPaMesh = wPa_tp1(aMesh, bMesh)
    wPbMesh = wPb_tp1(aMesh, bMesh)

    ucon_vars = solve_ucon(cMesh, dMesh, vMesh, w_tp1, utility, par, wPaMesh, wPbMesh, grids, verbose)
    con_vars = solve_con(cMesh, dMesh, vMesh, w_tp1, utility, grids, par, verbose)
    dcon_vars = solve_dcon(cMesh, dMesh, vMesh, w_tp1,  wPaMesh, wPbMesh, utility, grids, par, verbose)
    acon_vars = solve_acon(cMesh, dMesh, vMesh, w_tp1, wPb_tp1, utility, grids, par, verbose)

    # Post-decision grids
    A_acon = mMesh - cMesh - dMesh
    B_acon = nMesh + dMesh + utility.g(dMesh)

    # Policy functions on common grid
    cFunc = stablevalue2d(mGrid, nGrid, cMesh)
    dFunc = stablevalue2d(mGrid, nGrid, dMesh)

    # Value function on common grid
    vFunc = stablevalue2d(mGrid, nGrid, vMesh)

    # Value function derivatives on common grid
    vPmMesh = utility.P(cMesh)
    vPnMesh = par.DiscFac*wPb_tp1(A_acon, B_acon)

    vPmFunc = stablevalue2d(mGrid, nGrid, vPmMesh)
    vPnFunc = stablevalue2d(mGrid, nGrid, vPnMesh)

    return WorkingSolution(aMesh, bMesh,
                           mMesh, nMesh,
                           cMesh, dMesh,
                           vMesh,
                           vPmMesh, vPnMesh,
                           cFunc, dFunc,
                           vFunc,
                           vPmFunc, vPnFunc,
                           # notice that these should only be conditionally saved once all of the acon stuff gets sorted out
                           ucon_vars, con_vars,
                           dcon_vars, acon_vars)

def solve_ucon(C, D, V, w_tp1, utility, par, wPaMesh, wPbMesh, grids, verbose):
    if verbose:
        print("... solving unconstrained segment")

    # ====================
    # - grids and meshes -
    # ====================
    mGrid, nGrid = grids.mGrid, grids.nGrid

    aGrid, bGrid = grids.aGrid, grids.bGrid
    aMesh, bMesh = grids.aMesh, grids.bMesh

    # shape of common grid
    grid_shape = (len(mGrid), len(nGrid))

    # FOCs
    cMesh_ucon = utility.P_inv(par.DiscFac*wPaMesh)
    dMesh_ucon = (par.chi*wPbMesh)/(wPaMesh-wPbMesh) - 1

    # Endogenous grids
    mMesh_ucon = aMesh + cMesh_ucon + dMesh_ucon
    nMesh_ucon = bMesh - dMesh_ucon + utility.g(dMesh_ucon)

    # Value at endogenous grids
    vMesh_ucon = utility.u(cMesh_ucon) - par.alpha + par.DiscFac*w_tp1(aGrid, bGrid)

    cleanSegment((cMesh_ucon, dMesh_ucon, mMesh_ucon, nMesh_ucon), vMesh_ucon)

    # an indicator array that says if segment is optimal at
    # common grid points
    best_ucon = numpy.zeros(grid_shape, dtype = bool)
    segmentUpperEnvelope(grids, C, D, V, mMesh_ucon, nMesh_ucon, cMesh_ucon, dMesh_ucon, vMesh_ucon, best_ucon)
    vucon = numpy.copy(V)

    return (cMesh_ucon, dMesh_ucon, vMesh_ucon, mMesh_ucon, nMesh_ucon, best_ucon, vucon)

def solve_con(C, D, V, w_tp1, utility, grids, par, verbose):
    # ==============
    # -    con     -
    # ==============
    # FIXME this whole constrained thing can be upper-enveloped much easier as
    # we basically control (m, n) so we can just directly check it on the grid!
    if verbose:
        print("... solving fully constrained segment")

    # shape of common grid
    grid_shape = (len(grids.mGrid), len(grids.nGrid))

    # No FOCs, fully constrained -> consume what you have and doposit nothing
    cMesh_con = grids.mMesh   # this is fully constrained => m = c + a + b    = c
    dMesh_con = 0.0*cMesh_con # this is fully constraied  => n = b - d - g(d) = b

    # Endogenous grids (well, exogenous...)
    mMesh_con = grids.mMesh
    nMesh_con = grids.nMesh
    mGrid_con = grids.mGrid
    nGrid_con = grids.nGrid

    # aGrid_con = mGrid_con*0 # just zeros of the right shape
    # bGrid_con = nGrid_con # d = 0
    aMesh_con = mGrid_con*0 # just zeros of the right shape
    bMesh_con = nGrid_con # d = 0
    # Value at endogenous grids
    # vMesh_con = utility.u(cMesh_con) - alpha + DiscFac*w(aGrid_con, bGrid_con)
    vMesh_con = utility.u(cMesh_con) - par.alpha + par.DiscFac*w_tp1(aMesh_con, bMesh_con)

    # an indicator array that says if segment is optimal at
    # common grid points
    best_con = numpy.zeros(grid_shape, dtype = bool)
    segmentUpperEnvelope(grids,
                         C, D, V,
                         mMesh_con, nMesh_con,
                         cMesh_con, dMesh_con,
                         vMesh_con,
                         best_con)
    v_con = numpy.copy(V)
    return (cMesh_con, dMesh_con, vMesh_con, mMesh_con, nMesh_con, best_con, v_con)

def solve_dcon(C, D, V, w_tp1, wPaMesh, wPbMesh, utility, grids, par, verbose):
    if verbose:
        print("... solving deposit constrained segment")


    aMesh, bMesh = grids.aMesh, grids.bMesh

    # shape of common grid
    grid_shape = (len(grids.mGrid), len(grids.nGrid))

    # FOC for c, d constrained
    # ------------------------
    cMesh_dcon =  utility.P_inv(par.DiscFac*wPaMesh) # save as ucon
    # This preserves nans from cMesh
    dMesh_dcon = 0.0*cMesh_dcon

    # Endogenous grids (n-grid is exogenous)
    # --------------------------------------
    mMesh_dcon = aMesh + cMesh_dcon # + dMesh_dcon which is 0
    nMesh_dcon = bMesh

    # Value at endogenous grids
    vMesh_dcon = utility.u(cMesh_dcon) - par.alpha + par.DiscFac*w_tp1(aMesh, bMesh)

    cleanSegment((cMesh_dcon, dMesh_dcon, mMesh_dcon, nMesh_dcon), vMesh_dcon)

    # an indicator array that says if segment is optimal at
    # common grid points
    best_dcon = numpy.zeros(grid_shape, dtype=bool)
    segmentUpperEnvelope(grids,
                         C, D, V,
                         mMesh_dcon, nMesh_dcon,
                         cMesh_dcon, dMesh_dcon, vMesh_dcon,
                         best_dcon)
    v_dcon = numpy.copy(V)

    return (cMesh_dcon, dMesh_dcon, vMesh_dcon, mMesh_dcon, nMesh_dcon, best_dcon, v_dcon)

def solve_acon(C, D, V, w_tp1, wPb_tp1, utility, grids, par, verbose):
    if verbose:
        print("... solving post-decision cash-on-hand constrained segment")

    # shape of common grid
    grid_shape = (len(grids.mGrid), len(grids.nGrid))

    mGrid = grids.mGrid
    bGrid = grids.bGrid

    Na_acon = len(grids.aGrid)*4

    aGrid_acon = numpy.zeros(Na_acon)
    bGrid_acon = nonlinspace(0, bGrid.max(), Na_acon)
    aMesh_acon, bMesh_acon = numpy.meshgrid(aGrid_acon, bGrid_acon, indexing='ij')

    # Separate evaluation of wPb for acon due to constrained a = 0.0
    # --------------------------------------------------------------
    # We don't want wPbGrd_acon[0] to be a row, we want it to be the first
    # element, so we ravel
    wPbGrid_acon = wPb_tp1(aGrid_acon, bGrid_acon)
    wPbMesh_acon = wPb_tp1(aMesh_acon, bMesh_acon)

    # No FOC for c; fix at some "interesting" values
    # the approach requires b to be fixed. I'm not 100% I'm following the paper's
    # apparoch here, will make an inquiry with authors. The idea is to use
    #     uPc =         DiscFac*wPb(0, b)(1+chi/(1+d))
    #     c   = uPc_inv(DiscFac*wPb(0, b)(1+chi/(1+d)))
    # since a = 0. We then look for a sensible range of c's by evalutating on a
    # b-grid, and substituting in a very large d (so 1+d in the denominator makes
    # fraction vanish) or a very small d (d=0) such that the fraction becomes chi/1

    # [RFC] can this be done in a better way?
    CMin = utility.P_inv(par.DiscFac*wPbGrid_acon*(1 + par.chi))
    CMax = utility.P_inv(par.DiscFac*wPbGrid_acon*(1 + par.chi/(1+mGrid.max())))
    C_acon = numpy.array([nonlinspace(CMin[bi], CMax[bi], Na_acon) for bi in range(len(bGrid_acon))])

    # FOC for d constrained
    # the d's that fullfil FOC at the bottom of p. 93 follow from gPd(d) = chi/(1+d)
    # 0 = uPc(c)+DiscFac*wPb(a,b)(1+chi/(1+d))
    D_acon = par.chi/(utility.P(C_acon)/(par.DiscFac*wPbMesh_acon) - 1) - 1

    # Endogenous grids
    M_acon = aMesh_acon + C_acon + D_acon
    N_acon = bMesh_acon - D_acon - utility.g(D_acon)

    # value at endogenous grids
    V_acon = utility.u(C_acon) - par.alpha + par.DiscFac*w_tp1(aMesh_acon, bMesh_acon)

    cleanSegment((C_acon, D_acon, M_acon, N_acon), V_acon)

    # # an indicator array that says if segment is optimal at
    # # common grid points
    best_acon = numpy.zeros(grid_shape, dtype = bool)
    segmentUpperEnvelope(grids, C, D, V, M_acon, N_acon, C_acon, D_acon, V_acon, best_acon)
    VCopy = numpy.copy(V)

    return (C_acon, D_acon, V_acon, M_acon, N_acon, best_acon, VCopy, CMin, CMax)

def solve_period_retirement(rs_tp1, par, grids, utility, stablevalue):
    """
    Solves a period of the retirement model using EGM.

    Parameters
    ----------
    rs_tp1 : RetirementSolution
        holds the retirement solution for the next period
    par : MCNABParameters
        object that holds parameters relevant to utility and transitions
    grids : Grids
        holds all the grids and meshes for (m, n) and (a, b) space
    utility : Utility
        holds utility functions, inverses, derivatives, ...
    stablevalue : StableValue1D
        is used to create a stable interpolant of value functions

    Returns
    -------
    rs : RetirementSolution
        holds the retirement solution for the current period
    """
    # next period policy for retired agents
    c1d_tp1 = rs_tp1.c1d
    # next period value function for retired agents
    v_tp1 = rs_tp1.v

    # grids (for solution)
    aGrid = grids.aGrid_ret

    # ====================== #
    # - interior solutions - #
    # ====================== #
    m_tp1 = par.Ra*aGrid + par.y
    CGrid_tp1 = c1d_tp1(m_tp1)
    cP_tp1 = utility.P(CGrid_tp1)
    CGridInterior = utility.P_inv(par.DiscFac*par.Ra*cP_tp1)

    # ======================================== #
    # - corner solution (consume everything) - #
    # ======================================== #
    nInterior = math.ceil(len(aGrid)*0.2)
    mCorner = nonlinspace(1e-6, CGridInterior[0], nInterior)
    cCorner = numpy.copy(mCorner)

    # ===================== #
    # - combine solutions - #
    # ===================== #
    CGrid = numpy.concatenate((cCorner, CGridInterior))
    m = numpy.concatenate((mCorner, aGrid + CGridInterior))
    a = numpy.concatenate((numpy.full((nInterior,), 0.0), aGrid))

    # ======================= #
    # - policy interpolants - #
    # ======================= #
    c1DFunc = LinearInterp(m, CGrid)
    cFunc = lambda m, n: c1DFunc(m + n)

    # ========================================== #
    # - value (incl. derivatives) interpolants - #
    # ========================================== #
    vGrid = utility.u(CGrid) + par.DiscFac*v_tp1(a*par.Ra+par.y, a*0)
    # level
    v1DFunc = stablevalue(m, vGrid)
    vFunc = lambda m, n: v1DFunc(m+n)

    # derivatives
    vPm1DFunc = stablevalue(m, utility.P(CGrid))
    vPmFunc = lambda m, n: vPm1DFunc(m+n)
    vPnFunc = vPmFunc

    return RetirementSolution(c1DFunc, cFunc, v1DFunc, vFunc, vPmFunc, vPnFunc)