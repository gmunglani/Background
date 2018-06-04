import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# Create animation of background subtraction
def analysis_plot(X1, Y1, stack, work_out_path):
    """This functions plots the background analysis animation"""
    def data(i, stack, line):
        ax1.clear()
        line1 = ax1.plot_surface(stack.X, stack.Y, stack.im_medianf[:, :, i], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax1.set_title("{} Frame: {}".format(stack.val, i + 1))
        ax1.set_zlim(0, np.amax(stack.im_medianf))
        ax1.grid(False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        ax2.clear()
        minmax = np.ptp(np.ravel(stack.im_backf[:,:,i]))
        line2 = ax2.plot_surface(stack.X, stack.Y, stack.im_backf[:, :, i], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax2.set_title("Min to Max: {}".format(minmax))
        ax2.set_zlim(0, np.amax(stack.im_medianf))
        ax2.grid(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        ax3.clear()
        line3 = ax3.plot_surface(X1, Y1, stack.im_framef[i, :, :], cmap=cm.bwr, linewidth=0, antialiased=False)
        ax3.set_title("Number of Tiles: {}".format(stack.labelsf[:,i].size))
        ax3.set_zlim(0, np.amax(stack.im_medianf))
        ax3.grid(False)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])

        ax4.clear()
        signal = stack.labelsf[:,i]
        signal[signal > 0] = 0
        ax4.set_title("Number of Tiles with Signal: {}".format(-np.sum(signal)))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_zlim(0, 1)
        ax4.grid(False)
        ax4.set_xlabel('Variance', labelpad=10)
        ax4.set_ylabel('Skewness', labelpad=10)
        ax4.set_zlabel('Median', labelpad=10)
        varn = stack.propf[:,:,i]
        xyz = varn[stack.maskf[:,i]]
        xyz2 = varn[[not i for i in stack.maskf[:,i]]]
        line4 = ax4.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:, 3], c='blue')
        line4 = ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 3], c='red', s=80)

        line = [line1, line2, line3, line4]
        return line,

    # Define figures, axis and initialize
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    ax2 = fig.add_subplot(2,2,2,projection='3d')
    ax3 = fig.add_subplot(2,2,3,projection='3d')
    ax4 = fig.add_subplot(2,2,4,projection='3d')

    ax1.view_init(elev=15., azim=30.)
    ax2.view_init(elev=15., azim=30.)
    ax3.view_init(elev=15., azim=30.)
    ax4.view_init(elev=30., azim=210.)

    line1 = ax1.plot_surface(stack.X,stack.Y,stack.im_medianf[:,:,0],cmap=cm.bwr)
    line2 = ax2.plot_surface(stack.X,stack.Y,stack.im_backf[:,:,0],cmap=cm.bwr)
    line3 = ax3.plot_surface(X1,Y1,stack.im_framef[0,:,:],cmap=cm.bwr)
    line4 = ax4.scatter(10, 10, 10, c='red')

    line = [line1, line2, line3, line4]

    # Set up animation
    anim = animation.FuncAnimation(fig, data, fargs=(stack,line),frames=stack.range, interval=20, blit=False, repeat_delay=1000)

    pylab.rc('font', family='serif', size=10)
    plt.show()

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(extra_args=['-r', '25'])
   # anim.save(work_path + fname + '_' + val + '_back' + '.avi', writer=writer)

def logit(name):
    """ Logging input values """
    logger = logging.getLogger('back')
    hdlr = logging.FileHandler(name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(20)

    return logger