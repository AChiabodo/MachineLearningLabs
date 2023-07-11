import util
import DimensionalityReduction as dr
import plotting as plot

if __name__ == '__main__':
    DTR , LTR = util.load_iris()
    #DTR , LTR = util.load_digits()
    W = dr.LDA(DTR,LTR)
    plot.plot_scatter(W,LTR)
    #plot.plot_hists(W,LTR)
    #plot.pair_plot(DTR,LTR)