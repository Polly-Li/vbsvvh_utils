# read root file
# write histograms 
import ROOT
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patheffects import withStroke

import argparse
import os
from utils.common_utils import create_folder

def get_content(tfile):
    """
    Returns dict of content {'dir':[(obj_name, obj_cycle, obj_type, obj),...]}
    """
    result = {'main': []}

    # --- Loop over top-level keys ---
    for key in tfile.GetListOfKeys():
        name = key.GetName()
        cycle = key.GetCycle()
        obj_type = key.GetClassName()
        obj = key.ReadObj()

        if isinstance(obj, ROOT.TDirectory):
            # --- It's a subdirectory ---
            sub_list = []
            for subkey in obj.GetListOfKeys():
                subname = subkey.GetName()
                subcycle = subkey.GetCycle()
                subclass = subkey.GetClassName()
                subobj = subkey.ReadObj()
                sub_list.append((subname, subcycle, subclass, subobj))
            result[name] = sub_list
        else:
            # --- Object stored at top level ---
            result['main'].append((name, cycle, obj_type, obj))

    return result

def draw_2d_with_text(input_data, title, outputname, var1=None, min1=None, max1=None, var2=None, min2=None, max2=None,
                   cmap="viridis", fmt=".2f", fontsize=1.5, text_color="white"):
    """
    Draw a 2D map with imshow + colorbar + bin content text overlay.

    Parameters
    ----------
    data : 2D numpy array
        The map or histogram content.
    title : str
        Title of the plot and colorbar label.
    outputname : str
        Filename for saving (e.g., 'output.png').
    var1, var2 : str
        Axis variable names (x and y labels).
    min1, max1, min2, max2 : float
        Axis ranges corresponding to the data.
    cmap : str, optional
        Colormap name (default: 'viridis').
    fmt : str, optional
        Format string for bin text (default: '.2f').
    fontsize : int, optional
        Text font size (default: 8).
    text_color : str, optional
        Text color (default: 'white').
    """
    data = input_data[0]
    nx, ny = data.shape  #throw away error
    fontsize_auto = fontsize*50/max(nx,ny)
    data = np.transpose(data)

    xlabel = var1 if var1 is not None else 'x'
    ylabel = var2 if var2 is not None else 'y'
    xmin = min1 if min1 is not None else 0
    xmax = max1 if max1 is not None else nx
    ymin = min2 if min2 is not None else 0
    ymax = max2 if max2 is not None else ny
    
    # --- Draw image ---
    plt.figure()
    im = plt.imshow(data, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # --- Compute bin centers ---
    data = np.transpose(data)
    xbins = np.linspace(xmin, xmax, nx + 1)
    ybins = np.linspace(ymin, ymax, ny + 1)
    xcenters = 0.5 * (xbins[:-1] + xbins[1:])
    ycenters = 0.5 * (ybins[:-1] + ybins[1:])

    # --- Annotate bin values ---
    for ix, x in enumerate(xcenters):
        for iy, y in enumerate(ycenters):
            val = data[ix, iy]
            if np.isnan(val):  # skip NaN
                continue
            plt.text(
                x, y, format(val, fmt),
                color=text_color, ha="center", va="center", fontsize=fontsize_auto,
                path_effects=[withStroke(linewidth=0.5, foreground="black")] 
            )

    # --- Save and close ---
    plt.savefig(outputname, bbox_inches='tight', dpi=300)
    plt.close()

def draw_th1_barchart(data, title, outputname, var=None, minx=None, maxx=None,
                      color="skyblue", alpha=0.8):
    """
    Draw TH1D-style histogram as a bar chart (no error bars).
    content : np.array of bin contents
    """
    content = data[0] #throw away error
    nbins = len(content)
    xmin = minx if minx is not None else 0
    xmax = maxx if maxx is not None else nbins
    xbins = np.linspace(xmin, xmax, nbins + 1)
    xcenters = 0.5 * (xbins[:-1] + xbins[1:])

    plt.figure()
    plt.bar(xcenters, content, width=(xbins[1]-xbins[0]), color=color, alpha=alpha, edgecolor='black')
    plt.title(title)
    plt.xlabel(var if var is not None else 'x')
    plt.ylabel('Entries')
    plt.xlim(xmin, xmax)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputname, dpi=300)
    plt.close()

def draw_1d(data, title, outputname, var=None, minx=None, maxx=None,
                              marker='o', color='blue', capsize=1):
    """
    Draw TH1D-style histogram as points with vertical error bars.
    content, error : np.array of same length
    """
    content,error =data
    nbins = len(content)
    xmin = minx if minx is not None else 0
    xmax = maxx if maxx is not None else nbins
    xbins = np.linspace(xmin, xmax, nbins + 1)
    xcenters = 0.5 * (xbins[:-1] + xbins[1:])

    plt.figure()
    plt.errorbar(xcenters, content, yerr=error, fmt=marker, color=color, capsize=capsize)
    plt.title(title)
    plt.xlabel(var if var is not None else 'x')
    plt.ylabel('Entries')
    plt.xlim(xmin, xmax)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputname, dpi=300)
    plt.close()

def draw_tgraph(data, title, outputname, varx=None, vary=None,
                linestyle='-', marker='o', color='darkred'):
    """
    Draw TGraph-style XY plot (points connected by line).
    """
    x,y = data
    plt.figure()
    plt.plot(x, y, linestyle=linestyle, marker=marker, color=color)
    plt.title(title)
    plt.xlabel(varx if varx is not None else 'x')
    plt.ylabel(vary if vary is not None else 'y')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputname, dpi=300)
    plt.close()

def draw_object(obj,obj_type,output_name,title_input=None):
    title = title_input if title_input is not None else obj_type
    if obj_type.startswith('TH1') or obj_type.startswith('th1'):
        draw_1d(obj,title,output_name)
    elif obj_type.startswith('TH2') or obj_type.startswith('th2'):
        draw_2d_with_text(obj,title,output_name)
    elif obj_type.startswith('TGraph') or obj_type.startswith('tgraph'):
        draw_tgraph(obj,title,output_name)
    elif obj_type.startswith('TProfile') or obj_type.startswith('tprofile'):
        draw_1d(obj,title,output_name)
    else:
        raise TypeError(f"in draw object, unsupported object type: {obj_type}")


def convert_root_to_np(obj, obj_type):
    """convert object to np array"""
    def th1_to_np(hist):
        nbins = hist.GetNbinsX()
        content = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)], dtype=float)
        error   = np.array([hist.GetBinError(i)   for i in range(1, nbins + 1)], dtype=float)
        return content, error

    def th2_to_np(hist):
        nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
        arr = np.zeros((nx, ny), dtype=float)
        err = np.zeros((nx, ny), dtype=float)
        for ix in range(1, nx + 1):
            for iy in range(1, ny + 1):
                arr[ix - 1, iy - 1] = hist.GetBinContent(ix, iy)
                err[ix - 1, iy - 1] = hist.GetBinError(ix, iy)
        return arr,err

    def tgraph_to_np(graph):
        n = graph.GetN()
        x = np.array([graph.GetPointX(i) for i in range(n)], dtype=float)
        y = np.array([graph.GetPointY(i) for i in range(n)], dtype=float)
        return x, y

    def tprofile_to_np(profile):
        return th1_to_np(profile)

    obj_type = obj_type.lower()
    if obj_type.startswith('TH1') or obj_type.startswith('th1'):
        return th1_to_np(obj)
    elif obj_type.startswith('TH2') or obj_type.startswith('th2'):
        return th2_to_np(obj)
    elif obj_type.startswith('TGraph') or obj_type.startswith('tgraph'):
        return tgraph_to_np(obj)
    elif obj_type.startswith('TProfile') or obj_type.startswith('tprofile'):
        return tprofile_to_np(obj)
    else:
        raise TypeError(f"Unsupported ROOT object type: {obj_type}")

def main(input_file,output_dir):
    tfile_in = ROOT.TFile(input_file, "READ")
    hist_dict = get_content(tfile_in) 
    for folder in hist_dict.keys():
        create_folder(os.path.join(output_dir,folder))

    for folder,folder_content in hist_dict.items():
        if folder == 'ABCD': continue
        print(folder)
        for obj in folder_content:
            print(f'    {obj[:3]}') # obj is (name, cycle, obj_type, obj)
            name = obj[0] if obj[1]==1 else f'{obj[0]}_{obj[1]}'
            output_name = f'{output_dir}/{folder}/{name}.png'
            converted_data = convert_root_to_np(obj[3],obj[2])
            draw_object(converted_data,obj[2],output_name,title_input=obj[0])


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='plot all th1, tprofile, th2 and tgraphs in a root file')
    parser.add_argument("input", help = "input root file")
    parser.add_argument("--outdir",default=None, help = "output dir. will be the same as input if not provided")
    args = parser.parse_args()
    input_root = args.input
    output_dir = args.outdir if args.outdir is not None else os.path.join(os.path.dirname(os.path.abspath(input_root)),os.path.splitext(os.path.basename(input_root))[0]) # folder same name as input root if not provided
    main(input_root,output_dir)