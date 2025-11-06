import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Color maps

clr_map = {
    'EWK': "#ff8800",
    'Other': "#ffee00",
    'QCD': "#a3d42f",
    'ST': "#00dbbd",
    'WJets': "#5476a8",
    'ZJets': "#8C00FF",
    'ttbar': "#ff82ff",
    'ttx': "#00db6e",
}
sig_clr = "red"
data_clr = "blue"

def load_json(file_name):
    """Load yields JSON file"""
    with open(file_name) as f:
        return json.load(f)

def draw_barchart(sig, data, bkgs, save_name="cutflow.png", logy=False,sig_multiplier=None):
    """
    Draw stacked background bars + line-only signal and data
    sig, data, bkgs = dict structured like json["yields"]
    """
    regions = list(sig.keys())
    bkg_procs = list(clr_map.keys())

    fig, ax = plt.subplots(figsize=(10,6))
    if logy:
        ax.set_yscale("log")

    x = np.arange(len(regions))
    bar_width = 0.6

    for i, region in enumerate(regions):
        # stack backgrounds
        bottom = 0
        for proc in bkg_procs:
            val = bkgs[region].get(proc, [0,0])[0]
            ax.bar(i, val, bottom=bottom, color=clr_map[proc], width=bar_width,
                   label=proc if i==0 else "")
            bottom += val

        # signal (line only)
        sig_label = f"Signal x{sig_multiplier}" if sig_multiplier is not None else "Signal"
        sig_val = sig[region].get("Signal", [0,0])[0]
        ax.bar(i, sig_val, width=bar_width, edgecolor=sig_clr,
               fill=False, linewidth=2, label=sig_label if i==0 else "")

        # data (line only, dashed)
        if data is not None:
            data_val = data[region].get("Data", [0,0])[0]
            ax.bar(i, data_val, width=bar_width, edgecolor=data_clr,
                fill=False, linewidth=2, linestyle="--",
                label="Data" if i==0 else "")
            print("debug",region,bottom,data_val,sig_val)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel("Yield")
    ax.set_title("cutflow")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Draw stacked bar chart from yields JSON")
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("-o", "--output", default="cutflow.png", help="Output plot filename")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    parser.add_argument("--noall", action="store_true", help="Remove 'all_events' bar from plot")
    parser.add_argument("--unblind", action="store_true", help="Remove 'all_events' bar from plot")
    parser.add_argument("-x", "--sig_multiplier",type=float, default=1, help="sig scale")
    args = parser.parse_args()

    sig_multiplier = args.sig_multiplier if args.sig_multiplier is not None else 1

    data_all = load_json(args.input_json)["yields"]

    # Optionally remove "all_events"
    if args.noall and "all_events" in data_all:
        del data_all["all_events"]

    # Split into signal, data, bkg dicts
    sig = {
        r: {"Signal": np.array(v["Signal"], dtype=float) * sig_multiplier}
        for r, v in data_all.items()
    }
    if args.unblind:
        data = {r: {"Data": v["Data"]} for r, v in data_all.items()}
    else:
        data = None
    bkgs = {r: {k: val for k, val in v.items() if k in clr_map} for r, v in data_all.items()}

    draw_barchart(sig, data, bkgs, save_name=args.output, logy=args.logy, sig_multiplier=args.sig_multiplier)

if __name__ == "__main__":
    main()
