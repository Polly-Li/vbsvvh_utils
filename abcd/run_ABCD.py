import ROOT
import numpy as np
import csv
import os,sys
from itertools import combinations

from configs.rdf_project_name import get_project_filter
from configs.rdf_sample_names import sample_years # if not all year exist, will have to define explicitly
from configs.paths import ntuple_file, output_dir
from configs.paths import varlist_ABCD

from utils.common_utils import create_folder
from utils.rdf_handling import calculate_weight,apply_filters,read_root,get_weight
from utils.rdf_hists import book2D, bookHistogram
from utils.rdf_plots import set_drawing_options
from configs.read_varlist import create_cut
from configs.rdf_missing_var import fill_in_missing_cols
from configs.rdf_project_name import get_project_filter

ROOT.ROOT.EnableImplicitMT(8)

project_name = "c2v1p5_tight"

set_drawing_options()

def fitting_ABCD(ABCD,error_map,sig_region="A"):
    '''
    ABCD, error_map: dict containing BinContent and BinError of each region
    Return prediciton and propagated error of sig_region
    '''
    # Compute variance for each region
    variance = {key: (error_map[key] / ABCD[key])**2 if ABCD[key] != 0 else 0 for key in ABCD}
    variance_sum = sum(variance.values())
    # Perform prediction for region A only (as currently implemented)
    if sig_region == "A":
        if ABCD["D"] == 0:
            print("ABCD['D'] is zero, cannot divide by zero")
            prediction = 0
            error = 0
            return prediction, error
        prediction = ABCD["B"] * ABCD["C"] / ABCD["D"]
        error = prediction * ((variance_sum - variance["A"])**0.5)
        return prediction, error
    else:
        raise NotImplementedError(f"Prediction for sig_region '{sig_region}' is not implemented")

def get_ABCD(hist_ABCD):
    '''from a 2x2 hist2D, return BinContent and Error in dict
    regions defined as
    [[C, A],
    [D, B]]
    '''
    ABCD = {
        "A": hist_ABCD.GetBinContent(2, 2),
        "B": hist_ABCD.GetBinContent(2, 1),
        "C": hist_ABCD.GetBinContent(1, 2),
        "D": hist_ABCD.GetBinContent(1, 1)
    }

    error_map = {
        "A": hist_ABCD.GetBinError(2, 2),
        "B": hist_ABCD.GetBinError(2, 1),
        "C": hist_ABCD.GetBinError(1, 2),
        "D": hist_ABCD.GetBinError(1, 1)
    }
    return ABCD,error_map

def ABCD_result(hist_ABCD,sig_region="A"):
    '''preform ABCD calculation from a 2x2 hist2D, return prediciton,error,actual'''
    ABCD,error_map = get_ABCD(hist_ABCD)
    prediciton,pred_error = fitting_ABCD(ABCD,error_map,sig_region)
    actual = ABCD[sig_region]
    print(f"Prediction: {prediciton:.3f} pm {pred_error:.3f}, Actual: {actual:.3f}")
    act_err = error_map['A']
    return prediciton,pred_error,actual,act_err,ABCD

def hist2D_ABCD(df, x_var, x_cuts, y_var, y_cuts, weight="weight"):
    '''
    Create a 2x2 ABCD histogram using ROOT RDataFrame with bin edges.

    - df: ROOT RDataFrame
    - x_var, y_var: names of variables (strings)
    - x_cuts, y_cuts: [min, cut, max] defining bin edges
    - weight: weight column name
    '''

    if len(x_cuts) != 3 or len(y_cuts) != 3:
        raise ValueError("x_cuts and y_cuts must be of form [min, cut, max]")

    # Convert to numpy arrays (ROOT accepts these as C-style buffers)
    x_bins = np.array(x_cuts, dtype=np.double)
    y_bins = np.array(y_cuts, dtype=np.double)


    # Create TH2DModel with variable bin edges
    hist_ABCD_model = ROOT.ROOT.RDF.TH2DModel(
        f"ABCD_{x_var}_{y_var}",
        f"ABCD of {x_var} vs {y_var}",
        2, x_bins,
        2, y_bins
    )

    return df.Histo2D(hist_ABCD_model, x_var, y_var, weight)

def run_ABCD(root_files, years,sample_category,x_var, x_min, x_max, x_cut, y_var, y_min, y_max, y_cut, filename=None,write_hist=False, df_handling=True,extra_filter=None):
    df = {}
    filters = get_project_filter(project_name)
    histo2D = {}
    hist_ABCD = {}

    for year in years:
        df[year] = read_root(root_files[year,sample_category])
        #df[year] = df[year].Filter("HiggsScore > 0.5 && V1Score >0.5")
        if df_handling:
            df[year] = calculate_weight(df[year],sample_category)
            df[year] = fill_in_missing_cols(df[year])
            df[year] = apply_filters(df[year],filters)

        if extra_filter is not None:
            for f in extra_filter:
                df[year] = df[year].Filter(f)

    for year in years:
        # Book and fill 2D histograms
        histo2D[year] = book2D(df[year], x_var, [50, x_min, x_max], y_var, [50, y_min, y_max],f"ABCD of {x_var}-{y_var}").GetValue()
        hist_ABCD[year] = hist2D_ABCD(df[year], x_var, [x_min, x_cut, x_max], y_var, [y_min, y_cut, y_max]).GetValue()

    # Clone the first histogram as the starting point
    histo2D_all = histo2D[years[0]].Clone("histo2D_all")
    hist_ABCD_all = hist_ABCD[years[0]].Clone("hist_ABCD_all")

    # Add the rest
    for year in years[1:]:
        histo2D_all.Add(histo2D[year])
        hist_ABCD_all.Add(hist_ABCD[year])


    # Get prediction, error, actual from ABCD method
    correlation = histo2D_all.GetCorrelationFactor()
    pred, pred_err, act, act_err,ABCD = ABCD_result(hist_ABCD_all)
    total_error = (act_err**2+pred_err**2)**0.5

    if write_hist: #need a tfile to be already opened
        histo2D_all.Write(f'hist2d_{sample_category}_{x_var}_{y_var}')
        hist_ABCD_all.Write(f'ABCD_{sample_category}_{x_var}_{y_var}_{x_cut}_{y_cut}')

    # Plot
    if filename is not None:
        c = ROOT.TCanvas("c1", "c1", 800, 600)
        histo2D_all.Draw("COLZ")
        hist_ABCD_all.SetMarkerSize(1.5)
        hist_ABCD_all.Draw("TEXT SAME")

        # Draw cut lines
        line_x = ROOT.TLine(x_cut, y_min, x_cut, y_max)
        line_x.SetLineColor(ROOT.kOrange + 8)
        line_x.SetLineWidth(2)
        line_x.Draw()

        line_y = ROOT.TLine(x_min, y_cut, x_max, y_cut)
        line_y.SetLineColor(ROOT.kOrange + 8)
        line_y.SetLineWidth(2)
        line_y.Draw()

        # Annotate prediction in bin (2,2) of hist_ABCD (region A)
        x_center = (x_max+x_cut)/2/(x_max-x_min)
        y_center = (y_max+y_cut)/2
        y_text = (y_center - 0.025 * (y_max - y_min))/(y_max-y_min)  # slightly below center

        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextAlign(22)
        text.SetTextSize(0.03)
        text.DrawLatex(0.78,0.92,f"Prediction: {pred:.2f} \pm \sqrt ({pred_err:.2f}^2+ {act_err:.2f}^2)")
        #ptext.SetFillStyle(0)
        #ptext.SetLineWidth(0)
        text.Draw()

        # Save canvas
        c.Update()
        create_folder(filename)
        c.Print(f'{filename}.pdf')
        c.Close()

    return pred, pred_err, act, act_err, correlation, ABCD, total_error


def calculate_cut(var,varmin,varmax,ntuple_file,project_name):
    """calculate cut where half of sig is kept"""
    rootfiles = {
        'bkg':[ntuple_file[year,'bkg'] for year in sample_years],
        'sig':[ntuple_file[year,'sig'] for year in sample_years]
    }
    dfs = {
        sample:read_root(rootfiles[sample]) for sample in rootfiles.keys()
    }
    filters = get_project_filter(project_name)
    for sample_cat,df in dfs.items():
        dfs[sample_cat] = calculate_weight(dfs[sample_cat],sample_cat)
        dfs[sample_cat] = fill_in_missing_cols(dfs[sample_cat])
        dfs[sample_cat] = apply_filters(dfs[sample_cat],filters)
    yields = {
        sample: get_weight(dfs[sample]) for sample in rootfiles.keys()
    }
    if yields['sig']==0 or yields['bkg']==0:
        print('impossible to auto calculate cut as sig or bkg yield is 0. ')
        print(yields)
        sys.exit(1)

    else:
        temp_hist = bookHistogram(dfs['sig'],var,[50,varmin,varmax],'temp_hist',var).GetValue()
        target = 0.5 * yields['sig']
        cut = varmin
        for i in range(50):
            integral = temp_hist.Integral(0, i+1)
            if integral >=target:
                cut = varmin + (varmax-varmin)/50 * i 
                print(f'{var} cut defined at {cut}')
                return cut
        print('how can it not have a bin where the integral is more than half of the total??')
        return (varmax+varmin)/2

        



def main(project_name,varlist):
    cuts = get_project_filter(project_name)
    ranges = create_cut([],varlist)
    variables = ranges.keys()

    ABCD_out_dir = f"{output_dir}/cutbase_ABCD_result/"
    save_folder = os.path.join(ABCD_out_dir,project_name)
    output_csv = save_folder + "ABCD_predicitons.csv"
    create_folder(save_folder)
    output_root_file = save_folder+'cutbase_ABCD_hists.root'
    tfile_out = ROOT.TFile(output_root_file, "RECREATE")

    result = {}
    for var1,var2 in combinations(variables,2):
        min1,max1,cut1 =ranges[var1]
        min2,max2,cut2 =ranges[var2]
        if cut1 is None: 
            cut1 = calculate_cut(var1,min1,max1,ntuple_file,project_name)
            ranges[var1]=[min1,max1,cut1]
        if cut2 is None: 
            cut2 = calculate_cut(var2,min2,max2,ntuple_file,project_name)
            ranges[var2]=[min2,max2,cut2]
        print(var1,ranges[var1],var2,ranges[var2])
        pred_sig, pred_err_sig, act_sig,act_err_sig, correlation_sig, ABCD_sig,total_error_sig = run_ABCD(ntuple_file,sample_years,'sig',var1,min1,max1,cut1,var2,min2,max2,cut2, filename=f"{save_folder}/{var1}_{var2}_sig",write_hist=True,extra_filter=cuts,df_handling=True)
        # (ntuple_file,sample_years,'sig',var1,min1,max1,cut1,var2,min2,max2,cut2, filename=f"{save_folder}/{var1}_{var2}_sig",write_hist=True,extra_filter=cuts,df_handling=True)
        pred_bkg, pred_err_bkg, act_bkg,act_err_bkg, correlation_bkg, ABCD_bkg,total_error_bkg = run_ABCD(ntuple_file,sample_years,'bkg',var1,min1,max1,cut1,var2,min2,max2,cut2, filename=f"{save_folder}/{var1}_{var2}_bkg",write_hist=True,extra_filter=cuts,df_handling=True)

        contaminations = {key:ABCD_sig[key]/ABCD_bkg[key] if ABCD_bkg[key] != 0 else 0 for key in ABCD_bkg.keys()}
        signal_contamination = (contaminations['B']+contaminations['C']-contaminations['D'])/contaminations['A'] if contaminations['A'] !=0 else 0
        diff_bkg = abs(pred_bkg-act_bkg)
        total_error_bar = (act_err_bkg+ pred_err_bkg)
        diff_errbar = diff_bkg/total_error_bar if total_error_bar != 0 else 0
        result[var1,var2] = [act_sig,act_err_sig, correlation_bkg,act_bkg, pred_bkg,diff_bkg,act_err_bkg, pred_err_bkg,total_error_bkg, diff_errbar,signal_contamination]
    
    sorted_result = sorted(result.items(), key=lambda x: abs(x[1][2]), reverse=True) # sort against correlation

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['var1','var2','actual_D_sig','sig_err','correlation_bkg', 'actual_D_bkg', 'prediction_D_bkg','diff_bkg', 'err_bkg','pred_err_bkg','total_err_bkg', 'diff_errorbar','signal_contamination'])
        
        for (var1, var2), values in sorted_result:
            writer.writerow([var1, var2] + values)
    
if __name__ == "__main__":

    main(project_name,varlist_ABCD)