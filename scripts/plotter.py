from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rc

font_legend = fm.FontProperties(family = 'Times New Roman')
tnr = 'Times New Roman'
font = {'family': tnr}

rc('font', **font)

del fm.weight_dict['roman']
# fm._rebuild()

str_num_var = "Number of variables"

def get_time (file, time_measure='nano'):
    lines = [list(map(int, line.split())) for line in file.split("\n")[:-1]]
    nv = [line[0] for line in lines]
    time_us = [line[1] for line in lines]
    if time_measure == 'nano': 
        time_us = [nano / 10**9 for nano in time_us]
    elif time_measure == 'micro':
        time_us = [micro / 10**6 for micro in time_us]
    return nv, time_us
    

def plot_comparison(num_threads, gate = "vanilla"):
    loga = open(f"hyperplonk/logalk {gate} threads {num_threads}.txt").read()
    plk = open(f"hyperplonk/plk {gate} threads {num_threads}.txt").read()
    
    loga_nv, loga_time = get_time (loga, time_measure='micro')
    plk_nv, plk_time = get_time (plk, time_measure='micro')
    plt.clf()
    plt.title(f"HyperPlonk with lookup proof generation - {num_threads} thread(s)", **font, fontweight='bold')
    plt.ylabel("Time (seconds)", **font)  #micro before
    plt.xlabel(str_num_var, **font)
    plt.xticks(fontname=tnr)
    plt.yticks(fontname=tnr)
    plt.plot(loga_nv, loga_time, label="Logalookup")
    plt.plot(plk_nv, plk_time, label="Plookup")
    plt.legend(prop=font_legend)
    
    plt.savefig(f"plot-{gate}-compare-snark-{num_threads}-threads.png")

def plot_multithread(threads, lk = "logalk", gate = "vanilla"):
    plt.clf()
    lk_str = "Logalookup" if lk == "logalk" else "Plookup"
    plt.title(f"HyperPlonk with {lk_str} proof generation - Multithread", **font, fontweight='bold')
    plt.ylabel("Time (seconds)", **font)   #micro before
    plt.xlabel(str_num_var, **font)
    plt.xticks(fontname=tnr)
    plt.yticks(fontname=tnr)
    for nt in threads:
        _file = open(f"hyperplonk/{lk} {gate} threads {nt}.txt").read()
        nv, time = get_time (_file, time_measure='micro')
        plt.plot(nv, time, label=f"{nt} threads")
    plt.legend(prop=font_legend)
    plt.savefig(f"plot-multithread-{lk}.png")

def plot_comparison_lookup_subroutines(num_threads):
    loga = open(f"subroutines/logalk {num_threads} threads.txt").read()
    plk = open(f"subroutines/plk {num_threads} threads.txt").read()
    
    loga_nv, loga_time = get_time (loga, time_measure='nano')
    plk_nv, plk_time = get_time (plk, time_measure='nano')
    plt.clf()
    plt.title(f"Lookup proof generation - {num_threads} thread(s)", **font, fontweight='bold')
    plt.ylabel("Time (seconds)", **font)   #nano before
    plt.xlabel(str_num_var, **font)
    plt.xticks(fontname=tnr)
    plt.yticks(fontname=tnr)
    plt.plot(loga_nv, loga_time, label="Logalookup")
    plt.plot(plk_nv, plk_time, label="Plookup")
    plt.legend(prop=font_legend)
    
    plt.savefig(f"plot-compare-lookup-{num_threads}-threads.png")

def plot_multithread_lookup_subroutine(threads, lk = "logalk"):
    plt.clf()
    lk_str = "Logalookup" if lk == "logalk" else "Plookup"
    plt.title(f"{lk_str} subroutine proof generation - Multithread", **font, fontweight='bold')
    plt.ylabel("Time (seconds)", **font)  #nano before
    plt.xlabel(str_num_var, **font)
    plt.xticks(fontname=tnr)
    plt.yticks(fontname=tnr)
    for nt in threads:
        _file = open(f"subroutines/{lk} {nt} threads.txt").read()
        nv, time = get_time (_file, time_measure='nano')
        plt.plot(nv, time, label=f"{nt} threads")
    plt.legend(prop=font_legend)
    plt.savefig(f"plot-multithread-{lk}-subroutine.png")

for nt in [1,2,4,8,16]:
    plot_comparison(nt, gate='vanilla')
for nt in [1,2,4,8,16]:
    plot_comparison(nt, gate='jellyfish')
plot_multithread([1,2,4,8])
for nt in [1,2,4,8,16]:
   plot_comparison_lookup_subroutines(nt)
plot_multithread_lookup_subroutine([1,2,4,8])

