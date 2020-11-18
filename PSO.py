# Plot particle history as animation
import numpy as np
from sko.PSO import PSO
from CART_RF_PSO.TWB_RF import PSO_optimize_func


def opt_func(x):
    x1, x2, x3, x4, x5 = x
    x1 = int(round(x1 * 11)+1)
    x2 = int(round(x2 * 19)+1)
    x3 = int(round(x3 * 60)+40)
    x4 = int(round(x4 * 99)+1)
    x5 = int(round(x5 * 18)+2)
    print((x1, x2, x3, x4, x5))
    return PSO_optimize_func(x1, x2, x3, x4, x5)


pso = PSO(func=opt_func, dim=5, pop=20, max_iter=10, lb=[0, 0, 0, 0, 0], ub=[1, 1, 1, 1, 1])
pso.record_mode = True
pso.run()
print('best_x is ', list(pso.gbest_x))

# # %% Now Plot the animation
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# record_value = pso.record_value
# X_list, V_list = record_value['X'], record_value['V']
#
# fig, ax = plt.subplots(1, 1)
# ax.set_title('title', loc='center')
# line = ax.plot([], [], 'b.')
#
# X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
# Z_grid = demo_func((X_grid, Y_grid))
# ax.contour(X_grid, Y_grid, Z_grid, 20)
#
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
#
# plt.ion()
# p = plt.show()
#
#
# def update_scatter(frame):
#     i, j = frame // 10, frame % 10
#     ax.set_title('iter = ' + str(i))
#     X_tmp = X_list[i] + V_list[i] * j / 10.0
#     plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
#     return line
#
#
# ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=300)
# plt.show()
#
# # ani.save('pso.gif', writer='pillow')