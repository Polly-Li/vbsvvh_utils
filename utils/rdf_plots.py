#plots and styles
import ROOT
from configs.rdf_sample_names import get_sample_colour

def set_drawing_options():
    #disable statbox and skip drawing display
    ROOT.gROOT.SetBatch(1)
    ROOT.gStyle.SetOptStat(0)

def legend_setstyle(legend, font_size, x1, y1, x2, y2):
    """Set the legend style and enforce NDC (Normalized Device Coordinates), transparent background"""
    
    legend.SetNDC(True)  # Enforce NDC coordinates
    legend.SetFillStyle(4000)
    legend.SetTextSize(font_size)
    ROOT.gStyle.SetLegendFont(22)

    # Set position using NDC coordinates
    legend.SetX1NDC(x1)
    legend.SetY1NDC(y1)
    legend.SetX2NDC(x2)
    legend.SetY2NDC(y2)

    return legend

def line_hist_style(h,color,x_title,y_title='Event',x_unit='',y_unit=''):
    '''line style hist'''
    h.SetLineWidth(3)
    h.SetLineColor(color)
    h.GetXaxis().SetTitle(x_title+"/"+x_unit) if x_unit != '' else h.GetXaxis().SetTitle(x_title)
    h.GetYaxis().SetTitle(y_title+"/"+y_unit) if y_unit != '' else h.GetYaxis().SetTitle(y_title)
    h.GetXaxis().SetTitleSize(0.04)
    h.GetYaxis().SetTitleSize(0.04)

def fill_hist_style(h,color,x_title,y_title='Event',x_unit='',y_unit=''):
    '''fill style hist'''
    h.SetFillStyle(1001)
    h.SetLineColor(color)
    h.SetFillColor(color)
    h.GetXaxis().SetTitle(x_title+"/"+x_unit) if x_unit != '' else h.GetXaxis().SetTitle(x_title)
    h.GetYaxis().SetTitle(y_title+"/"+y_unit) if y_unit != '' else h.GetYaxis().SetTitle(y_title)
    h.GetXaxis().SetTitleSize(0.04)
    h.GetYaxis().SetTitleSize(0.04)
    #h.GetXaxis().SetTitleSize(0.05)
    #h.GetYaxis().SetTitleSize(0.05)
    #h.SetTitleSize(0.3,'t')


def sig_hist_style(h,sample_type,x_title,y_title='Event',x_unit='',y_unit=''):
     h.SetLineStyle(1)
     h.SetLineColor(get_sample_colour(sample_type))
     h.SetLineWidth(3)
     h.GetXaxis().SetTitle(x_title+"("+x_unit+")")
     h.GetYaxis().SetTitle(y_title+"("+y_unit+")")
     h.GetXaxis().SetTitleSize(0.4)
     h.GetYaxis().SetTitleSize(0.4)

     
def bkg_hist_style(h,sample_type,x_title,y_title='Event',x_unit='',y_unit=''):
     h.SetFillStyle(1001)
     h.SetLineColor(get_sample_colour(sample_type))
     h.SetFillColor(get_sample_colour(sample_type))
     h.GetXaxis().SetTitle(x_title+"("+x_unit+")")
     h.GetYaxis().SetTitle(y_title+"("+y_unit+")")
     h.GetXaxis().SetTitleSize(0.05)
     h.GetYaxis().SetTitleSize(0.05)
     h.SetTitleSize(0.3,'t')

def data_hist_style(h,sample_type,x_title,y_title='Event',x_unit='',y_unit=''):
     h.SetLineStyle(1)
     h.SetLineColor(get_sample_colour(sample_type))
     h.SetLineWidth(3)
     h.GetXaxis().SetTitle(x_title+"("+x_unit+")")
     h.GetYaxis().SetTitle(y_title+"("+y_unit+")")
     h.GetXaxis().SetTitleSize(0.05)
     h.GetYaxis().SetTitleSize(0.05)
     h.SetTitleSize(0.3,'t')

def hist_to_jpg(hist, file_name, need_statbox=False, handle_overflow=True, draw_opt="HIST"):
    '''simple function to draw hist as "HIST" to an image'''
    import os, ROOT
    # Ensure ROOT doesn't open windows
    ROOT.gROOT.SetBatch(True)
    # Hide statbox if not needed
    if not need_statbox:
        ROOT.gStyle.SetOptStat(0)
    else:
        ROOT.gStyle.SetOptStat(1)

    # Check and create the directory
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # Warn if file exists
    if os.path.exists(file_name):
        print(f"Warning: '{file_name}' already exists and will be overwritten.")

    # Handle overflow and underflow bins
    if ((not need_statbox) and handle_overflow):
        nbins = hist.GetNbinsX()
        # Add underflow to first bin
        hist.SetBinContent(1, hist.GetBinContent(1) + hist.GetBinContent(0))
        hist.SetBinError(1, (hist.GetBinError(1)**2 + hist.GetBinError(0)**2)**0.5)
        # Add overflow to last bin
        hist.SetBinContent(nbins, hist.GetBinContent(nbins) + hist.GetBinContent(nbins + 1))
        hist.SetBinError(nbins, (hist.GetBinError(nbins)**2 + hist.GetBinError(nbins + 1)**2)**0.5)

    # Create canvas and draw
    c = ROOT.TCanvas("c", "c", 600, 600)
    hist.Draw(draw_opt)


    c.Update()
    c.Print(file_name)

    # Clean up
    c.Close()

def c_to_jpg(canvas, file_name):
    """draw a canvas to a jpg"""
    import os
    # Check and create the directory
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # Warn if file exists
    if os.path.exists(file_name):
        print(f"Warning: '{file_name}' already exists and will be overwritten.")

    canvas.Update()
    canvas.Print(file_name)
