import numpy as np
import matplotlib.pyplot as plt
import gsa

def ishigami_single_repetition(n_xi, n_eta):
    """Calculate indices using stochastic ishigami function
    [0] is standard, [1] is vard
    vi shape is [k, 2, dshape]
    """

    gsa.settings(n_xi=n_xi, n_eta=n_eta)
    input_deck = gsa.generate_samples(returnDeck=True)

    qoi, sig_sq = gsa.stochastic_ishigami_function(input_deck.matrix["A"])
    fA = {'qoi': qoi, 'sig_sq': sig_sq}
    qoi, sig_sq = gsa.stochastic_ishigami_function(input_deck.matrix["B"])
    fB = {'qoi': qoi, 'sig_sq': sig_sq}
    datashape = (3,) + qoi.shape
    qoi = np.zeros(datashape)
    sig_sq = qoi.copy()
    for param_id in range(input_deck.k):
        AB = input_deck.matrix["A"].copy()
        AB[:, param_id] = input_deck.matrix["B"][:, param_id].copy()
        qoi[param_id], sig_sq[param_id] = gsa.analytic_ishigami_function(AB)

    results = {'fA': fA['qoi'], 'fB': fB['qoi'], 'fAB': qoi, 'sigsq_A': fA['sig_sq'], 'sigsq_B': fB['sig_sq'], 'sigsq_AB': sig_sq}
    # pass results into function that computes var, si, ti, vi, eni

    var, vi, eni, si, ti = gsa.compute_saltelli_indices(results)
    qoi = np.mean(np.concatenate((fA['qoi'][:],fB['qoi'][:]),axis=0),axis=0)

    return var, vi, eni, si, ti, qoi


def draw_plot(data, offset, edge_color, ax, fill_color="white"):
    pos = np.arange(data.shape[1])+offset+1
    bplot = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True)
    for element in ['boxes', 'medians']:
        for item in bplot[element]:
            item.set(color=edge_color, linewidth=2)
    for element in ['whiskers', 'caps']:
        for item in bplot[element]:
            item.set(color=edge_color, linewidth=2)
    for item in bplot['boxes']:
        item.set(facecolor=fill_color)
    for item in bplot['fliers']:
        item.set(marker='o', markeredgecolor=edge_color, alpha=0.5)
    return bplot


def run_ishigami_example(n_eta, n_xi, n_reps, figname=False):
    qoi = np.zeros(n_reps)
    var = np.zeros((n_reps, 2))
    vi = np.zeros((n_reps, 3, 2))
    eni = vi.copy(); si = vi.copy(); ti = vi.copy()

    for rep in range(n_reps):
        var[rep], vi[rep], eni[rep], si[rep], ti[rep], qoi[rep] = ishigami_single_repetition(n_xi, n_eta)

    
    fig, ax = plt.subplots()
    # Analytic
    for i in range(3):
        x = np.array([-1, 1])*0.5 + i + 1
        y = np.ones(2) * a_si[i]
        anplot = ax.plot(x, y, '*-', color='black', markersize=10)
    # Numerical
    A = si[:,:,0]
    B = si[:,:,1]
    stanplot = draw_plot(A, -0.2, "red", ax)
    vardplot = draw_plot(B, +0.2, "skyblue", ax)
    plt.xticks([1,2,3], [1,2,3])
    ax.legend([anplot[0], stanplot['boxes'][0], vardplot['boxes'][0]], ['Analytic', 'Standard', 'Var dec.'], loc='upper right')
    plt.xlabel("Input number")
    plt.ylabel("First-order Sobol' index")
    ax.set_title(r'$N_{\eta}=$' + str(n_eta) + r'$, \,\, N_{\xi}=$' + str(n_xi) + r'$, \,\, N_{reps}=$' + str(n_reps))
    ax.set_ylim([-0.5, 1])
    if figname:
        # don't know why but the working directory is the parent directory instead of GSA_code
        # this will make the figure save, but when there's time I'd like to know why that's happening
        plt.savefig('GSA_code/ishigami_figures/eta{0}_xi{1}_reps{2}.png'.format(n_eta, n_xi, n_reps))



# Instantiate gsa input_deck in gsa.functions
k=3
for i in range(k):
    gsa.parameter(key="x"+str(i+1), pmin=-np.pi, pmax=np.pi, dist="uniform")

# Analytic results
a_var, a_vi, a_eni, a_si, a_ti, a_qoi = gsa.analytic_ishigami_indices()


# Run repetitions of ishigami function and generate boxplot of results
# run_ishigami_example(n_eta=5, n_xi=100, n_reps=1000, figname=True)            # baseline
# run_ishigami_example(n_eta=10, n_xi=100, n_reps=1000, figname=True)           # n_eta * 2
# run_ishigami_example(n_eta=5, n_xi=200, n_reps=1000, figname=True)            # n_xi * 2
# run_ishigami_example(n_eta=50, n_xi=100, n_reps=1000, figname=True)           # n_eta * 10
# run_ishigami_example(n_eta=5, n_xi=1000, n_reps=1000, figname=True)            # n_xi * 10
# 
# run_ishigami_example(n_eta=5, n_xi=100, n_reps=100, figname=True)            # baseline
# run_ishigami_example(n_eta=10, n_xi=100, n_reps=100, figname=True)           # n_eta * 2
# run_ishigami_example(n_eta=5, n_xi=200, n_reps=100, figname=True)            # n_xi * 2
# run_ishigami_example(n_eta=50, n_xi=100, n_reps=100, figname=True)           # n_eta * 10
# run_ishigami_example(n_eta=5, n_xi=1000, n_reps=100, figname=True)            # n_xi * 10
run_ishigami_example(n_eta=50, n_xi=1000, n_reps=100, figname=True)


plt.show()