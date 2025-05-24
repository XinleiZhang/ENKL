if __name__ == '__main__':
    import os
    import numpy as np
    import fluidfoam as ff
    import tools
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cbook, cm
    from matplotlib.colors import LightSource

    ##############################################################################
    foamCase = os.path.join(tools.dafiCase, 'results_ensemble',
                            f'sample_{tools.sampleId}')
    U = ff.readvector(foamCase, f'{tools.timeDir:g}', 'U')
    theta1 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'theta1')
    theta2 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'theta2')
    theta3 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'theta3')
    theta4 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'theta4')
    theta5 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'theta5')
    theta6 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'theta6')
    # g1 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'g1')
    # g2 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'g2')
    # g3 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'g3')
    # g4 = ff.readscalar(foamCase, f'{tools.timeDir:g}', 'g4')
    print('Ux, min, max', U[0, :].min(), U[0, :].max())
    print('Uy, min, max', U[1, :].min(), U[1, :].max())
    print('Uz, min, max', U[2, :].min(), U[2, :].max())
    print('theta1, min, max', theta1.min(), theta1.max())
    print('theta2, min, max', theta2.min(), theta2.max())
    print('theta3, min, max', theta3.min(), theta3.max())
    print('theta4, min, max', theta4.min(), theta4.max())
    print('theta5, min, max', theta5.min(), theta5.max())
    print('theta6, min, max', theta6.min(), theta6.max())
    x, y, z = ff.readmesh(foamCase)

    theta = np.concatenate(
        (theta1.reshape(-1, 1), theta2.reshape(-1, 1), theta3.reshape(-1, 1),
         theta4.reshape(-1, 1), theta5.reshape(-1, 1), theta6.reshape(-1, 1)),
        axis=1)

    # g = np.concatenate((g1.reshape(-1, 1), g2.reshape(-1, 1), g3.reshape(
    #     -1, 1), g4.reshape(-1, 1)),
    #                    axis=1)

    np.savetxt(f'theta-range-{tools.sampleId}.csv',
               np.concatenate((theta.min(axis=0).reshape(
                   -1, 1), theta.max(axis=0).reshape(-1, 1)),
                              axis=1),
               delimiter=',')

    #############################################################################
    theta_range = np.loadtxt(f'theta-range-{tools.sampleId}.csv',
                             delimiter=',')
    # print(theta_range)
    # log mesh
    a1 = 0
    a2 = 1
    base = 10
    numPoints = 50
    t1_stencil = (np.logspace(a1, a2, numPoints, base=10) -
                  1) * (theta_range[0][1] - theta_range[0][0]) / (np.power(
                      base, a2) - np.power(base, a1)) + theta_range[0][0]
    a1 = 0
    a2 = 1
    base = 10
    numPoints = 50
    t2_stencil = np.flip((np.logspace(a1, a2, numPoints, base=10) - 1) *
                         (theta_range[1][0] - theta_range[1][1]) /
                         (np.power(base, a2) - np.power(base, a1)) +
                         theta_range[1][1])
    a1 = 0
    a2 = 1
    base = 1000
    numPoints = 50
    t5_stencil = np.flip((np.logspace(a1, a2, numPoints, base=base) - 1) *
                         (theta_range[4][0] - theta_range[4][1]) /
                         (np.power(base, a2) - np.power(base, a1)) +
                         theta_range[4][1])
    # print(t1_stencil_log)
    # print(t2_stencil_log)
    # print(t5_stencil_log)

    t1_stencil = np.linspace(theta_range[0][0], theta_range[0][1], numPoints)
    t2_stencil = np.linspace(theta_range[1][0], theta_range[1][1], numPoints)
    t5_stencil = np.linspace(theta_range[4][0], theta_range[4][1], numPoints)

    ##########################################################################
    # create data used for symbolic regression
    theta1_mesh, theta2_mesh, theta5_mesh = np.meshgrid(
        t1_stencil, t2_stencil, t5_stencil)
    theta_table = np.concatenate(
        (theta1_mesh.reshape(-1, 1), theta2_mesh.reshape(
            -1, 1), np.zeros_like(theta2_mesh).reshape(-1, 1),
         np.zeros_like(theta2_mesh).reshape(-1, 1), theta5_mesh.reshape(
             -1, 1), np.zeros_like(theta2_mesh).reshape(-1, 1)),
        axis=1)
    #############################################################################
    # calculation

    g1_by_Shih = tools.g1_Shih(theta1_mesh[:, :, 0], theta2_mesh[:, :, 0])
    g1_by_nn = tools.g1_nn(theta_table).reshape(numPoints, numPoints,
                                                numPoints)
    g1_by_nn_best = np.zeros_like(theta5_mesh[:, :, 0])
    theta5_surface = np.zeros_like(theta5_mesh[:, :, 0])
    for i in range(len(t1_stencil)):
        for j in range(len(t2_stencil)):
            id = np.argmin(np.abs(g1_by_nn[i, j, :] - g1_by_Shih[i, j]))
            theta5_surface[i, j] = theta5_mesh[i, j, id]
            g1_by_nn_best[i, j] = g1_by_nn[i, j, id]
    #############################################################################
    # plot style
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "Helvetica"
        'figure.dpi': 800,
        'axes.labelsize': 6,
        'xtick.direction': 'in',
        'xtick.labelsize': 6,
        'xtick.top': True,
        'xtick.major.width': 0.5,
        'ytick.direction': 'in',
        'ytick.labelsize': 6,
        'ytick.right': True,
        'ytick.major.width': 0.5,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'axes.titlesize': 6,
        'grid.linewidth': 0.5
    })
    plotStyleFlowfield = {'linestyle': '-', 'color': 'brown', 'linewidth': 1}

    # plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    z_depth = np.min(-np.power(t1_stencil, 2) / 2)
    ls = LightSource(270, 45)
    # ax = plt.axes()
    ax.plot_surface(theta1_mesh[:, :, 0],
                    theta2_mesh[:, :, 0],
                    theta5_surface,
                    facecolor='gray',
                    edgecolor='black',
                    lw=0.5,
                    rstride=1,
                    cstride=1,
                    alpha=0.5)

    ax.plot_surface(theta1_mesh[:, :, 0] + 10,
                    theta2_mesh[:, :, 0],
                    0.5 * theta1_mesh[:, :, 0] * theta2_mesh[:, :, 0],
                    facecolor='gray',
                    edgecolor='black',
                    lw=0.5,
                    rstride=1,
                    cstride=1,
                    alpha=0.5)

    ax.plot(t1_stencil + 10, -t1_stencil, -np.power(t1_stencil, 2) / 2,
            **plotStyleFlowfield)

    # ax.plot(t1_stencil + 10, -t1_stencil, z_depth * np.ones_like(t1_stencil),
    #         **plotStyleFlowfield)
    # plot contour
    levels = np.linspace(-0.5, -0.05, 21)
    ax.contourf(theta1_mesh[:, :, 0] + 10,
                theta2_mesh[:, :, 0],
                g1_by_Shih,
                levels=levels,
                zdir='z',
                offset=z_depth * 1.05,
                cmap='viridis')
    ax.contourf(theta1_mesh[:, :, 0],
                theta2_mesh[:, :, 0],
                g1_by_nn_best,
                levels=levels,
                zdir='z',
                offset=z_depth * 1.05,
                cmap='viridis')
    ax.set(xlabel=r'$\theta_1$',
           ylabel=r'$\theta_2$',
           zlabel=r'$\theta_5$',
           xlim=(0, 17.5),
           ylim=(-10, 0),
           zlim=(z_depth, 0))
    ax.view_init(elev=20, azim=-70, roll=0)
    plt.gca().set_box_aspect((7, 4, 3))
    plt.savefig('show.png')
    plt.show()
