import numpy as np
from scipy.optimize import minimize

# import gtsam

def calc_huber_norm(center, points, norms, bias=0, huber_radius=0.4):
    """Calculate the normal vector for a superpixel using huber loss on pixel
        out-of-plane components.  Optimizes for the normal vector which
        minimizes out-of-plane components of pixels wrt superpixel center and
        normal vector.  Uses Huber loss to be robust to outliers and initializes
        from mean of pixel norms.
        See eq4 of https://www.dropbox.com/s/h9bais2wnw1g9f0/root.pdf?dl=0
    Arguments:
        center: [x,y,z] of superpixel center
        points: nx3 array of pixel locations
        norms:  nx3 array of pixel norms (for initialization only)
        bias:   residual bias, i.e. Huber( norm.point + b )
        huber_radius:   radius of huber loss
    Returns:
        norm:   [nx, ny, nz] of normal vector
    """

    # initialize norm
    initNorm = np.mean(norms, axis=0)
    initNorm = initNorm / np.linalg.norm(initNorm)
    print(initNorm)
    
    # optimize
    points = points-center
    def f(norm):
        toRet = np.sum(
            np.square(
            np.maximum(
            np.minimum(
                norm.dot(points.transpose()), huber_radius), -huber_radius)))
        return toRet
    def c(norm):
        return np.linalg.norm(norm)-1
    constr = {'type': 'eq', 'fun': c}
    sol = minimize(f, initNorm, constraints=constr, tol=0.0001)
    return sol.x

    """
    # FG stuff for huber norm calculation
    graph = gtsam.NonlinearFactorGraph()
    norm = gtsam.symbol(ord('n'), 0)
    dots = []
    for i in range(np.size(points, 0)):
        dots.append(gtsam.symbol(ord('d'), i))
    # create noise models
    hard_constraint = gtsam.noiseModel_Constrained.All(1)
    huber_noise = gtsam.noiseModel_Robust(
                gtsam.noiseModel_mEstimator_Huber(huber_radius),
                gtsam.noiseModel_Isotropic.Sigma(1, 1000.0)) # 1D, cov=1 (arbitrary)
    # add factors
    points = points-center
    for i in range(len(points)):
        # dot product calculation
        print(np.reshape(points[i,:], (1,3)))
        print(np.dot(np.reshape(points[i,:], (1,3)),np.array([[1.],[1.],[1.]])))
        dotfactor = gtsam.JacobianFactor(
            norm, np.reshape(points[i,:], (1,3)),
            dots[i], -np.eye(1),
            np.array([0.]), hard_constraint)
        graph.add(gtsam.LinearContainerFactor(dotfactor))
        # try to minimize dot product (with huber loss for robustness)
        huberfactor = gtsam.PriorFactorVector(
            dots[i], np.array([-bias], dtype=np.float), huber_noise)
        graph.add(huberfactor)
    # add approximate unit constraint
    normFactor = gtsam.JacobianFactor(
        norm, np.array([[1.,1.,1.]]), np.array([1.]), hard_constraint
    )
    # initial guess
    initial = gtsam.Values()
    initial.insert(norm, initNorm)
    for i in range(np.size(points,0)):
        val = np.array([np.dot(initNorm, points[i,:])])
        initial.insert(dots[i], val)
    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    marginals = gtsam.Marginals(graph, result)
    for i,d in enumerate(dots):
        print("dots{} covariance:\n{}\n".format(i, marginals.marginalCovariance(d)))
    print("norm covariance:\n{}\n".format(marginals.marginalCovariance(norm)))

    return np.reshape(result.atVector(norm), 3)
    """

if __name__ == "__main__":
    pixelNorms = np.array([[1,0,0],[0,1,0],[1,0,0],[0.99,0.141,0]])
    centerLoc = np.array([1,2,0])
    pixelLocs = np.array([[1,3,0], [1,1,0], [1.1,5,5], [0.9,-1, -5], [10,1,0], [19,3,0]])
    # pixelNorms = np.array([[1,1,1.]])
    # centerLoc = np.array([0,0,0.])
    # pixelLocs = np.array([[1,0,0.],[0,1,0],[0,0,0.1]])
    expected_norm = np.array([1,0,0])

    norm = calc_huber_norm(centerLoc, pixelLocs, pixelNorms, bias=0,
        huber_radius=1)

    print(norm)