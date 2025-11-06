# because the old version is terrible

# build a root file that store output plot
# loss, bce, disco (line 1D)
# ROC (scatter)
# distribution by count/yield for score1, score2 (line 1D)
# dist by count, yield (2D)
# profX, proY for count/yield (line 1D)
# pass count, yield, significance scan (line 1D)
# significance scan (2D)
# A,B,C,D, pred (2D)
# err +pred_err + total_err (2D)
# cut map (for convenience) (2D x2)
# mask map (2D)
# masked significance (2D)

import sys,argparse,os
import ROOT
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import json, yaml
from array import array
import copy

from configs.paths import double_training_result_path,output_dir
from configs.rdf_sample_names import sample_types, filter_sample_type

from utils.rdf_handling import decode_sample_code

from utils.rdf_handling import read_root, decode_sample_code
from utils.rdf_hists import book2D, bookHistogram, wrap_overflow2d
from combine_utils.run_combine import run_combine
from utils.common_utils import create_folder

def read_config(file_path):
    with open(file_path, 'r') as file:
        ABCD_output_dict = yaml.safe_load(file)
    return ABCD_output_dict

def create_TH1D(name, title, nbins, xmin, xmax):
    h = ROOT.TH1D(name, title, nbins, xmin, xmax)
    return h

def create_TH2D(name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax):
    h = ROOT.TH2D(name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax)
    return h

def rebin_ABCD(hist2D, nbinx, nbiny):
    """
    Rebin a 2D histogram into 2x2 regions (A, B, C, D) based on bin cuts.
    """

    # --- Safety checks ---
    nx = hist2D.GetNbinsX()
    ny = hist2D.GetNbinsY()

    # if not (2 <= nbinx <= nx - 1):
    #     raise ValueError(f"nbinx={nbinx} invalid, must be in [1, {nx-1}]")
    # if not (2 <= nbiny <= ny - 1):
    #     raise ValueError(f"nbiny={nbiny} invalid, must be in [1, {ny-1}]")

    # --- Get axis info ---
    xaxis = hist2D.GetXaxis()
    yaxis = hist2D.GetYaxis()

    x_low = xaxis.GetXmin()
    x_up  = xaxis.GetXmax()
    y_low = yaxis.GetXmin()
    y_up  = yaxis.GetXmax()

    x_cut = xaxis.GetBinLowEdge(nbinx)
    y_cut = yaxis.GetBinLowEdge(nbiny)

    # --- Define new bin edges ---
    xbins =  array('d', [x_low, x_cut, x_up])
    ybins =  array('d', [y_low, y_cut, y_up])

    # --- Create new histogram ---
    hname = hist2D.GetName() + f"_rebin_ABCD_{nbinx}_{nbiny}"
    htitle = hist2D.GetTitle() + f" (rebin @ x={nbinx}, y={nbiny})"
    h2_new = ROOT.TH2F(hname, htitle, 2, xbins, 2, ybins)
    h2_new.Sumw2()  # ensure error storage

    # --- Accumulate contents and errors ---
    for ix in range(1, nx + 1):
        xcenter = xaxis.GetBinCenter(ix)
        ix_new = 1 if xcenter < x_cut else 2

        for iy in range(1, ny + 1):
            ycenter = yaxis.GetBinCenter(iy)
            iy_new = 1 if ycenter < y_cut else 2

            val = hist2D.GetBinContent(ix, iy)
            err = hist2D.GetBinError(ix, iy)

            old_val = h2_new.GetBinContent(ix_new, iy_new)
            old_err = h2_new.GetBinError(ix_new, iy_new)

            h2_new.SetBinContent(ix_new, iy_new, old_val + val)
            h2_new.SetBinError(ix_new, iy_new, math.sqrt(old_err**2 + err**2))

    return h2_new

def get_ABCD(hist_ABCD):
    '''from a 2x2 hist2D, return BinContent and Error in dict
    return array [0,1,2,3] where 
     [[1,3]
      [0,2]]
    '''
    ABCD =[ 
        hist_ABCD.GetBinContent(1, 1),
        hist_ABCD.GetBinContent(1, 2),
        hist_ABCD.GetBinContent(2, 1),
        hist_ABCD.GetBinContent(2, 2)
    ]

    error_map = [
        hist_ABCD.GetBinError(1, 1),
        hist_ABCD.GetBinError(1, 2),
        hist_ABCD.GetBinError(2, 1),
        hist_ABCD.GetBinError(2, 2)
    ]

    return ABCD,error_map

def fitting_ABCD(ABCD,error_map):
    '''
    ABCD, error_map: dict containing BinContent and BinError of each region
    Return prediciton and error of A
    '''
    # Compute variance for each region
    if ABCD[0] == 0: return np.nan,np.nan 
    else:
        prediction = ABCD[1]*ABCD[2]/ABCD[0]
        variance = sum([(error_map[i] / ABCD[i])**2 if ABCD[i] != 0 else 0 for i in [0,1,2]])

        error = prediction * math.sqrt(variance)
        return prediction, error

def clone_hist(hist, name, title=None):
    """
    Clone an existing histogram and return an empty histogram (same binning and axes).
    """
    if not hist:
        raise ValueError("Input histogram is None.")

    # Clone structure
    new_title = title if title else hist.GetTitle()
    h_clone = hist.Clone(name)
    h_clone.SetTitle(new_title)
    h_clone.Reset()
    return h_clone

def fill_test_hist(hist, multiplier=1.0, error=0.1, include_flow=False):
    """
    for testing
    """
    nx = hist.GetNbinsX()
    ny = hist.GetNbinsY()

    # Range of bins to loop over
    xbins = range(0 if include_flow else 1, nx + 2 if include_flow else nx + 1)
    ybins = range(0 if include_flow else 1, ny + 2 if include_flow else ny + 1)

    for ix in xbins:
        for iy in ybins:
            hist.SetBinContent(ix, iy, multiplier*(ix+iy))
            hist.SetBinError(ix, iy, multiplier*error)

def safe_SetBinContent2D(hist,ix,iy,value):
    """ To only save valid value into histogram. Also handle when ix,iy is not recognized as int"""
    if math.isfinite(value):
        hist.SetBinContent(int(ix), int(iy), value)
    else:
        hist.SetBinContent(int(ix), int(iy), 0.0)  # or skip entirely

def get_cutmap(nbin1, min1, max1, nbin2, min2, max2):
    """ return any np array dim(nbinx,nbiny,2) that (x,y) gives cutx,cuty"""
    cut_map = np.zeros((nbin1,nbin2,2))
    for ix in range(nbin1):  
        for iy in range(nbin2):
            cut_map[ix,iy,: ] = [min1+(ix)*(max1-min1)/nbin1,min2+(iy)*(max2-min2)/nbin2]
    return cut_map

def convert_hist1D(arr, name, title=None, err=None):
    """
    Convert a 1D numpy array to a ROOT.TH1D histogram.
    """
    if title is None:
        title = name

    nbin = len(arr)
    hist = ROOT.TH1D(name, title, nbin, 0.5, nbin + 0.5)  # bins centered at 1,2,...

    for i, val in enumerate(arr, start=1):
        hist.SetBinContent(i, float(val))
        if err is not None:
            hist.SetBinError(i, float(err[i - 1]))
        else:
            hist.SetBinError(i,0)

    hist.SetDirectory(0)  # detach from gDirectory
    return hist

def convert_scatter(arr, name, title=None, err=None):
    """
    Convert a 2D numpy array (x,y[,errors]) to a ROOT.TGraphErrors.
    """
    if title is None:
        title = name

    npoints = arr.shape[0]
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)

    if err is None:
        graph = ROOT.TGraph(npoints, x, y)
    else:
        err = np.asarray(err)
        # if given shape (N,2): use as symmetric x/y error
        if err.shape[1] == 2:
            xerr = err[:, 0].astype(float)
            yerr = err[:, 1].astype(float)
            graph = ROOT.TGraphErrors(npoints, x, y, xerr, yerr)
        else:
            raise ValueError("err should be shape (N, 2): [x_err, y_err].")

    graph.SetName(name)
    graph.SetTitle(title)
    return graph

def true_positive_rate(df_sig,name,score_var,nbins=50, xmin=1, xmax=1):
    return positive_rate(df_sig,name,score_var,nbins, xmin, xmax)

def false_positive_rate(df_bkg,name,score_var,nbins=50, xmin=1, xmax=1):
    return positive_rate(df_bkg,name,score_var,nbins, xmin, xmax)
    
def positive_rate(df,name,score_var,nbins=50, xmin=1, xmax=1): 
    #if df is sig, then it is tpr; if bkg then fpr
    def calculate_cut_at(nbins,xmin,xmax):
        return [ i*((xmax-xmin)/nbins)+xmin for i in range(0,nbins+2,1)]
    
    pr_array = [] #to return

    h = df.Histo1D((name, score_var, nbins, xmin, xmax), score_var).GetValue()

    # absorb underflow into first bin
    h.SetBinContent(1, h.GetBinContent(0) + h.GetBinContent(1))
    h.SetBinError(1, (h.GetBinError(0)**2 + h.GetBinError(1)**2)**0.5)

    # absorb overflow into last bin
    last_bin = h.GetNbinsX()
    h.SetBinContent(last_bin, h.GetBinContent(last_bin) + h.GetBinContent(last_bin+1))
    h.SetBinError(last_bin, (h.GetBinError(last_bin)**2 + h.GetBinError(last_bin+1)**2)**0.5)
    h.SetBinContent(last_bin+1,0)
    h.SetBinError(last_bin+1, 0)


    total = h.Integral(0, nbins+1)  # include under/overflow
    if total <=0:
        print("no events found?")
        sys.exit(1)

    integral = total
    cuts = calculate_cut_at(nbins, xmin, xmax)

    for i in range(nbins+1):
        integral -= h.GetBinContent(i) # events >nbin threshold
        pr = integral / total 
        pr_array.append([cuts[i],pr,integral])

    # this gives last bin != 0 if score == 1 event exists
    # so here drops the last bin so the integral combine overflows into the 2nd last bin
    # pr_array[nbins] = [cuts[nbins],0.0,0.0]
    # removed because last bin is set to 0

    return pr_array, h

def profiles(df,var1,var2,nbin1,min1,max1,nbin2,min2,max2,weight='weight'):
    hist2D = book2D(df,var1,[nbin1,min1,max1],var2,[nbin2,min2,max2],f'{var1}-{var2}',weight).GetValue()
    hist2D = wrap_overflow2d(hist2D)

    profX_hist = hist2D.ProfileX("profX")
    profY_hist = hist2D.ProfileY("profY")
    profX = []
    profY = []
    profX_err = []
    profY_err = []
    for i in range(1,nbin1):
        profX.append(profX_hist.GetBinContent(i))
        profX_err.append(profX_hist.GetBinError(i))
    for i in range(1,nbin2):
        profY.append(profY_hist.GetBinContent(i))
        profY_err.append(profY_hist.GetBinError(i))

    return profX,profX_err,profY,profY_err,profX_hist,profY_hist

def main(ML_name):
    #var setting
    var1 = "abcdnet_score1"
    var2 = "abcdnet_score2"
    nbin1, min1, max1 = 50, 0, 1
    nbin2, min2, max2 = 50, 0, 1
    weight = 'weight'
    region_labels = ['A', 'B', 'C', 'D'] #ll,ul,lr,ur(sig) in root convention

    #read trained result
    config_json = ABCD_output_dict[ML_name]['config_json']
    learning_json = ABCD_output_dict[ML_name]['learning_json']
    output_epoch = ABCD_output_dict[ML_name]['output_epoch']
    result_dir = os.path.dirname(config_json) 
    
    with open(learning_json) as f:
        learning_dict = json.load(f)

    sig_files = [f'{result_dir}/output/cut_{year}_sig_abcdnet.root' for year in ['2016preVFP','2016postVFP','2017','2018']]
    bkg_files = [f'{result_dir}/output/cut_{year}_bkg_abcdnet.root' for year in ['2016preVFP','2016postVFP','2017','2018']]
    df_sig = read_root(sig_files)
    df_bkg = read_root(bkg_files)
    df_sig = decode_sample_code(df_sig)
    df_bkg = decode_sample_code(df_bkg)

    hist_sig_count = book2D(df_sig,var1,[nbin1, min1, max1],var2,[nbin2, min2, max2],'hsig_count',"signal scores",weight=None)
    hist_sig_yield = book2D(df_sig,var1,[nbin1, min1, max1],var2,[nbin2, min2, max2],'hsig_weight',"signal scores",weight=weight)
    hist_bkg_count = book2D(df_bkg,var1,[nbin1, min1, max1],var2,[nbin2, min2, max2],'hbkg_count',"background scores (by count)",weight=None)
    hist_bkg_yield = book2D(df_bkg,var1,[nbin1, min1, max1],var2,[nbin2, min2, max2],'hbkg_weight',"background scores (weighted)",weight=weight)
            
    
    output_root_file = f'{output_dir}/ABCDnet/result_{ML_name}.root'

    # Check if the folder already exists
    create_folder(output_root_file)
    tfile = ROOT.TFile(output_root_file, "RECREATE")

    #save distribution plots
    tfile.cd()
    tfile.mkdir('dist2D')
    tfile.cd('dist2D')
    hist_sig_count.Write()
    hist_bkg_count.Write()
    hist_sig_yield.Write()
    hist_bkg_yield.Write()

    #create holders
    input_hist_dict = {
        'sig':hist_sig_yield,
        'bkg':hist_bkg_yield,
    }
    hist_name = ([f'value_{i}' for i in region_labels]  # store ABCD value
                + [f'err_{i}' for i in region_labels]  # store ABCD error
                + ['pred', 'pred_err', 'total_err','valid_value']) #valid value store 1 if all values in ABCD is not nan
    h_dict = {
        'sig':{n:create_TH2D('sig_'+n,'sig_'+n,nbin1, min1, max1, nbin2, min2, max2) for n in hist_name},
        'bkg':{n:create_TH2D('bkg_'+n,'bkg_'+n,nbin1, min1, max1, nbin2, min2, max2) for n in hist_name},
        'common':{
            #from 2d dist
            'significance':create_TH2D('significance','significance',nbin1, min1, max1, nbin2, min2, max2),
            'closure_mask':create_TH2D('closure_mask','closure_mask',nbin1, min1, max1, nbin2, min2, max2),
            'masked_significance':create_TH2D('masked_significance','masked_significance',nbin1, min1, max1, nbin2, min2, max2),
            'bkg_closure_diff':create_TH2D('bkg_closure_diff','bkg_closure_diff',nbin1, min1, max1, nbin2, min2, max2),
            'relative_closure':create_TH2D('relative_closure','relative_closure',nbin1, min1, max1, nbin2, min2, max2), #abs(pred/act)/error
            'limit':create_TH2D('limit','limit(sig,bkg,obs=bkg)',nbin1, min1, max1, nbin2, min2, max2),
            'masked_limit':create_TH2D('masked_limit','limit(sig,bkg,obs=bkg)',nbin1, min1, max1, nbin2, min2, max2),
        },
        'loss':{
            # for loss in learning config
        },
        'ROC':{
            #var1, var2
        },
        'dist1d':{ 
            #dist1d for count, yield for var1, var2
            #proX proY for count, yield, for var1, var2
            #pass count, yield, significance scan
        },
        'per_sample_type':{ 
            #distribution per sample_type
        }
    }
    result_template = {
        'ABCD_map':np.zeros((nbin1, nbin2, 4)),
        'err_map':np.zeros((nbin1, nbin2, 4)),
        'pred_map':np.zeros((nbin1, nbin2)),
        'pred_err_map':np.zeros((nbin1, nbin2)),
        'total_err_map':np.zeros((nbin1, nbin2)),
        'closure_diff':np.zeros((nbin1, nbin2)),
    }
    result = {
        'sig': copy.deepcopy(result_template),
        'bkg': copy.deepcopy(result_template),
    }
    cut_map = get_cutmap(nbin1, min1, max1, nbin2, min2, max2)
    
    #plot per sample type
    tfile.cd()
    tfile.mkdir('per_sample_type')
    tfile.cd('per_sample_type')
    for sig_type in sample_types['sig']:
        h_dict['per_sample_type'][(sig_type,'count')] = book2D(filter_sample_type(df_sig,sig_type),var1,[nbin1,min1,max1],var2,[nbin2,min2,max2],f'{sig_type}_count',f'{sig_type} by count',weight=None)
        h_dict['per_sample_type'][(sig_type,'yield')] = book2D(filter_sample_type(df_sig,sig_type),var1,[nbin1,min1,max1],var2,[nbin2,min2,max2],f'{sig_type}_yield',f'{sig_type} by yield',weight=weight)
        h_dict['per_sample_type'][(sig_type,'count')].Write()
        h_dict['per_sample_type'][(sig_type,'yield')].Write()

    for bkg_type in sample_types['bkg']:
        h_dict['per_sample_type'][(bkg_type,'count')] = book2D(filter_sample_type(df_bkg,bkg_type),var1,[nbin1,min1,max1],var2,[nbin2,min2,max2],f'{bkg_type}_count',f'{bkg_type} by count',weight=None)
        h_dict['per_sample_type'][(bkg_type,'yield')] = book2D(filter_sample_type(df_bkg,bkg_type),var1,[nbin1,min1,max1],var2,[nbin2,min2,max2],f'{bkg_type}_yield',f'{bkg_type} by yield',weight=weight)
        h_dict['per_sample_type'][(bkg_type,'count')].Write()
        h_dict['per_sample_type'][(bkg_type,'yield')].Write()


    #generate loss plots
    tfile.cd()
    tfile.mkdir('loss')
    tfile.cd('loss')
    for key,arr in learning_dict.items():
        h_dict['loss'][key] = convert_hist1D(arr,key)
        h_dict['loss'][key].Write()
    print("done loss")
        
    #calculate ROC
    tfile.cd()
    tfile.mkdir('ROC')
    tfile.cd('ROC')

    tpr1,tpr1_h = true_positive_rate(df_sig,'sig_tpr1',var1,nbin1,min1,max1)
    fpr1,fpr1_h = false_positive_rate(df_bkg,'bkg_fpr1',var1,nbin1,min1,max1)

    ROC1_x = [fpr1[i][1] for i in range(len(fpr1))]
    ROC1_y = [tpr1[i][1] for i in range(len(tpr1))]
    ROC1 = np.column_stack((ROC1_x, ROC1_y))
    ROC1_h = convert_scatter(ROC1,'ROC1',f'{var1} ROC')
    tpr1_h.Write('tpr1')
    fpr1_h.Write('fpr1')
    ROC1_h.Write('ROC1')

    tpr2,tpr2_h = true_positive_rate(df_sig,'sig_tpr2',var2,nbin2,min2,max2)
    fpr2,fpr2_h = false_positive_rate(df_bkg,'bkg_fpr2',var2,nbin2,min2,max2)

    ROC2_x = [fpr2[i][1] for i in range(len(fpr2))]
    ROC2_y = [tpr2[i][1] for i in range(len(tpr2))]
    ROC2 = np.column_stack((ROC2_x, ROC2_y))
    ROC2_h = convert_scatter(ROC2,'ROC2',f'{var2} ROC')

    tpr2_h.Write('tpr2')
    fpr2_h.Write('fpr2')
    ROC2_h.Write('ROC2')

    print("done ROC")

    #generated 1d dist plots
    range_dict = {
        var1: [nbin1, min1, max1],
        var2: [nbin2, min2, max2],
    }
    df_dict = {
        'sig':df_sig,
        'bkg':df_bkg
    }
    
    hist_1d = {}
    for var in [var1,var2]:
        tfile.cd()
        tfile.mkdir(var)
        tfile.cd(var)
        for sample in df_dict.keys():
            hist_1d[sample,var,'count'] = bookHistogram(df_dict[sample],var,range_dict[var],f'{sample}_{var}_count',f'{sample}_{var}_count',weight=None) 
            hist_1d[sample,var,'yield'] = bookHistogram(df_dict[sample],var,range_dict[var],f'{sample}_{var}_yield',f'{sample}_{var}_yield') 
            hist_1d[sample,var,'count'].Write()
            hist_1d[sample,var,'yield'].Write()

            hist_1d[sample,var,'count_pass'] = hist_1d[sample,var,'count'].GetCumulative(True,'_pass')
            hist_1d[sample,var,'yield_pass'] = hist_1d[sample,var,'yield'].GetCumulative(True,'_pass')
            hist_1d[sample,var,'count_pass'].Write()
            hist_1d[sample,var,'yield_pass'].Write()

    print("done 1d graphs")

    #profiles
    tfile.cd()
    tfile.mkdir('prof')
    tfile.cd('prof')
    total_h_dict = {
        'sig_count':hist_sig_count,
        'bkg_count':hist_bkg_count,
        'sig_yield':hist_sig_yield,
        'bkg_yield':hist_bkg_yield
        }
    profX_hist = {}
    profY_hist = {}
    for i,h in total_h_dict.items():
        profX_hist[h] = h.ProfileX(f"{i}_profX")
        profY_hist[h] = h.ProfileY(f"{i}_profY")
        profX_hist[h].Write()
        profY_hist[h].Write()
    print("done profile plots")

    #calculate ABCD result
    tfile.cd()
    tfile.mkdir('ABCD')
    for sample in input_hist_dict.keys():
        input_hist = input_hist_dict[sample]
        for ix in range(1, nbin1+1):  # loop from 2 to nbin1-1
            for iy in range(1, nbin2+1):
                
                #rebin to ABCD
                h_temp = rebin_ABCD(input_hist, ix, iy)
                tfile.cd('ABCD')
                h_temp.Write(f'{sample}_{ix}_{iy}') 

                #calculate prediction
                ABCD, err = get_ABCD(h_temp)
                pred, pred_err = fitting_ABCD(ABCD, err)
                total_error = err[3] + pred_err

                #fill result
                result[sample]['ABCD_map'][ix-1, iy-1, :] = ABCD
                result[sample]['err_map'][ix-1, iy-1, :] = err
                result[sample]['pred_map'][ix-1, iy-1] = pred
                result[sample]['pred_err_map'][ix-1, iy-1] = pred_err
                result[sample]['total_err_map'][ix-1, iy-1] = total_error
                result[sample]['closure_diff'][ix-1,iy-1] = abs(pred - ABCD[3])
                
                safe_SetBinContent2D(h_dict[sample][f'pred'],ix,iy,pred)
                safe_SetBinContent2D(h_dict[sample][f'pred_err'],ix,iy,pred_err)
                safe_SetBinContent2D(h_dict[sample][f'total_err'],ix,iy,total_error)

                if math.isfinite(pred) and math.isfinite(pred_err) and math.isfinite(total_error):
                    h_dict[sample][f'valid_value'].SetBinContent(ix,iy,1)
                else:
                    h_dict[sample][f'valid_value'].SetBinContent(ix,iy,0)
                    
                for i, region in enumerate(region_labels):
                    h_dict[sample][f'value_{region}'].SetBinContent(ix,iy,ABCD[i])
                    h_dict[sample][f'err_{region}'].SetBinContent(ix,iy,err[i])

                #prevent overwriting warning
                h_temp.Delete()
                del h_temp
        
        tfile.mkdir(sample)
        tfile.cd(sample)
        for name,h in h_dict[sample].items():
            h.Write()
    
    print("done ABCD result")
    tfile.cd()
    
    # --- Create mask ---
    closure_mask =  result['bkg']['closure_diff'] < result['bkg']['total_err_map']
    closure_mask_img = np.where(closure_mask,1,0)
    
    significance = result['sig']['ABCD_map'][:,:,3]/np.sqrt(result['bkg']['pred_map'])
    mask_signif = np.where(closure_mask,significance,0)
    relative_closure = result['bkg']['closure_diff']/result['bkg']['total_err_map']
    
    for ix in range(1, nbin1+1):  # loop from 2 to nbin1-1
        for iy in range(1, nbin2+1):
            h_dict['common']['closure_mask'].SetBinContent(ix,iy,closure_mask_img[ix-1,iy-1])
            safe_SetBinContent2D(h_dict['common']['significance'],ix,iy,significance[ix-1,iy-1])
            safe_SetBinContent2D(h_dict['common']['masked_significance'],ix,iy,mask_signif[ix-1,iy-1])
            safe_SetBinContent2D(h_dict['common']['bkg_closure_diff'],ix,iy,result['bkg']['closure_diff'][ix-1,iy-1])
            safe_SetBinContent2D(h_dict['common']['relative_closure'],ix,iy,relative_closure[ix-1,iy-1])

    for name,h in h_dict['common'].items():
        if name == 'limit':continue #because limit is not yet calculated
        h.Write()
    print("done closure map")

    #find the cut where signif is max within closure
    limit = np.zeros((nbin1, nbin2))

    idx1, idx2 =  np.unravel_index(np.argmax(mask_signif),result['bkg']['closure_diff'].shape)
    s = result['sig']['ABCD_map'][idx1,idx2,3]
    b = result['bkg']['pred_map'][idx1,idx2]
    limit_at_max_signif = run_combine(s,b,b,False)["m"]
    limit[idx1,idx2] = limit_at_max_signif
    safe_SetBinContent2D(h_dict['common']['limit'],idx1+1,idx2+1,limit_at_max_signif) #+1 because idx is in np convention, use safe setbin because idx somehow is not int. (-_-)
    if closure_mask_img[idx1,idx2]==1:
        safe_SetBinContent2D(h_dict['common']['masked_limit'],idx1+1,idx2+1,limit_at_max_signif) 
        
    print("max idx",idx1,idx2,cut_map[idx1,idx2,:])
    print(f'limit at max significance (s={s:.3f},b={b:.3f}) {limit_at_max_signif:.3f}')

    #coarse limit scan
    space = 5
    for ix in range(1, nbin1+1,space):  # using root convention for index
        for iy in range(1, nbin2+1,space):
            s = h_dict['sig']['value_D'].GetBinContent(ix,iy)
            b = h_dict['bkg']['pred'].GetBinContent(ix,iy)
            print(s,b)
            if s>0 and b>0:
                l = run_combine(s,b,b,False)["m"]
            else:
                l = -1
            limit[ix-1,iy-1] = l  # using root convention for index
            print(cut_map[ix-1,iy-1,:],'limit',l)
            h_dict['common']['limit'].SetBinContent(ix,iy,limit[ix-1,iy-1])

            is_closed = closure_mask_img[ix-1,iy-1]
            if is_closed ==1:
                h_dict['common']['masked_limit'].SetBinContent(ix,iy,limit[ix-1,iy-1])
            else:
                h_dict['common']['masked_limit'].SetBinContent(ix,iy,0)
                
    #scan near highest limit
    mask = (limit != -1) & (limit != 0)
    mask_limit = np.where(mask,limit,99)
    idx1, idx2 =  np.unravel_index(np.argmin(mask_limit),limit.shape)
    print("max limit idx",idx1,idx2,cut_map[idx1,idx2,:],limit[idx1,idx2])

    #switch idx to rdf convention
    idx1 +=1 
    idx2 +=1 
    for ix in range(idx1-space+1, idx1+space-1):  
        for iy in range(idx2-4, idx2+4,2):
            s = h_dict['sig']['value_D'].GetBinContent(ix,iy)
            b = h_dict['bkg']['pred'].GetBinContent(ix,iy)
            print(s,b)
            is_closed = closure_mask_img[ix-1,iy-1]
            if s>0 and b>0 and limit[ix-1,iy-1]==0 and is_closed:
                l = run_combine(s,b,b,False)["m"]
            elif limit[ix-1,iy-1]!=0:
                l =limit[ix-1,iy-1]
            else:
                l = -1
            limit[ix-1,iy-1] = l
            print(cut_map[ix-1,iy-1,:],'limit',l)
            h_dict['common']['limit'].SetBinContent(ix,iy,limit[ix-1,iy-1])

            is_closed = closure_mask_img[ix-1,iy-1]
            if is_closed ==1:
                h_dict['common']['masked_limit'].SetBinContent(ix,iy,limit[ix-1,iy-1])
            else:
                h_dict['common']['masked_limit'].SetBinContent(ix,iy,0)

    #scan near highest limit wwithin closure
    mask = (closure_mask) & (limit != -1) & (limit != 0)
    mask_limit = np.where(mask,limit,99)
    idx1, idx2 =  np.unravel_index(np.argmin(mask_limit),limit.shape)
    print("max limit idx in closure",idx1,idx2,cut_map[idx1,idx2,:],limit[idx1,idx2])
    
    #switch idx to rdf convention
    idx1 +=1 
    idx2 +=1 
    for ix in range(idx1-space+1, idx1+space-1):  
        for iy in range(idx2-space+1, idx2+space-1):
            s = h_dict['sig']['value_D'].GetBinContent(ix,iy)
            b = h_dict['bkg']['pred'].GetBinContent(ix,iy)
            print(s,b)
            is_closed = closure_mask_img[ix-1,iy-1]
            if s>0 and b>0 and limit[ix-1,iy-1]==0 and is_closed:
                l = run_combine(s,b,b,False)["m"]
            elif limit[ix-1,iy-1]!=0:
                l =limit[ix-1,iy-1]
            else:
                l = -1
            limit[ix-1,iy-1] = l
            print(cut_map[ix-1,iy-1,:],'limit',l)
            h_dict['common']['limit'].SetBinContent(ix,iy,limit[ix-1,iy-1])

            is_closed = closure_mask_img[ix-1,iy-1]
            if is_closed ==1:
                h_dict['common']['masked_limit'].SetBinContent(ix,iy,limit[ix-1,iy-1])
            else:
                h_dict['common']['masked_limit'].SetBinContent(ix,iy,0)

    h_dict['common']['limit'].Write()
    h_dict['common']['masked_limit'].Write()
    tfile.Close()
   
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name","-n",default='all', help = "all to run every report, else run the ML_output")
    args = parser.parse_args()
    ML_names = args.input_name
    ABCD_output_dict = read_config(double_training_result_path)
    
    if ML_names == 'all':
        for ML_name in ABCD_output_dict.keys():
            main(ML_name)
    else:
        main(ML_names)
