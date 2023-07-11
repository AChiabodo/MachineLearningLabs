from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
def plot_scatter(D : numpy.array, L : numpy.array):
    classes = numpy.unique(L)
    with PdfPages('target/scatter.pdf') as pdf:
        for dIdx1 in range(D.shape[0]):
            for dIdx2 in range(D.shape[0]):
                fig = plt.figure()
                if dIdx1 == dIdx2:
                    continue
                for i in range(classes.shape[0]):
                    plt.scatter(D[dIdx1, L == classes[i]], D[dIdx2, L == classes[i]], label=f"Class: {classes[i]}")
                plt.title(f"plot : {D.shape[0]}")
                plt.legend()
                plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
                pdf.savefig(fig)
                plt.close(fig)

def plot_hists(DTR: numpy.array, LTR: numpy.array):
    classes = numpy.unique(LTR)
    with PdfPages('target/hists.pdf') as pdf:
        for i in range(DTR.shape[0]):
            fig = plt.figure()
            plt.xlabel(f"Feature: {i}")
            plt.ylabel("Density")
            for j in range(classes.shape[0]):
                plt.hist(DTR[i, LTR == j], density=True, bins=20, label=f'Class: {classes[j]}', alpha=0.4)
            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)

def __pair_plot_hist(DTR: numpy.array, LTR: numpy.array, axis, i, j):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    axis[i, j].hist(f0[0, :], density=True, bins=20, label='Spoofed', alpha=0.4)
    axis[i, j].hist(f1[0, :], density=True, bins=20, label='Spoofed', alpha=0.4)


def __pair_plot_scatter(DTR: numpy.array, LTR: numpy.array, axis, i, j):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    axis[i, j].scatter(f0[0, :], f0[1, :], label='Spoofed')
    axis[i, j].scatter(f1[0, :], f1[1, :], label='Authentic')


def pair_plot(DTR: numpy.array, LTR: numpy.array):
    feature_count = DTR.shape[0]

    fig, axis = plt.subplots(nrows=feature_count, ncols=feature_count, squeeze=False)
    fig.set_size_inches(feature_count * 4, feature_count * 4)

    # Iterate through features to plot pairwise.
    for i in range(0, feature_count):
        for j in range(0, feature_count):
            if i == j:
                __pair_plot_hist(DTR[i:i + 1], LTR, axis, i, j)
            else:
                new_data = numpy.vstack([DTR[i:i+1], DTR[j:j+1]])
                __pair_plot_scatter(new_data, LTR, axis, i, j)

    plt.show()