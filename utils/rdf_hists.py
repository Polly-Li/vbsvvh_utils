import ROOT

def bookHistogram(df, variable, ranges, name, title,  weight = "weight"):
    '''return df histo1D with given variable'''
    #return histo. if var = sum_AK8_pt, need to filter
    [nbin, xlow, xup] = ranges[:3]
    if variable == "sum_AK8_pt":
        df = df.Filter("nGoodAK8>0")
    if "vbsj" in variable:
        df = df.Filter("vbs_idx_max_Mjj[0]!=-99") #has_vbs
    

    # ROOT::RDF::TH1DModel::TH1DModel	(	const char * 	name,
    # const char * 	title,
    # int 	nbinsx,
    # double 	xlow,
    # double 	xup 
    # )	
    hist_model = ROOT.ROOT.RDF.TH1DModel(name, title, nbin, xlow, xup)
    # Return the histogram object
    
    if weight is None:
        # call overload without weight
        return df.Histo1D(hist_model, variable)
    else:
        return df.Histo1D(hist_model, variable, weight)
      
def book2D(df, var1, range1_, var2, range2_, name, title=None, weight = 'weight'):
    """
    Books a 2D histogram from a ROOT RDataFrame.

    Parameters:
    - df: The ROOT RDataFrame to process.
    - var1: The variable for the X-axis.
    - range1_: A tuple (nbins, xmin, xmax) defining the X-axis binning.
    - var2: The variable for the Y-axis.
    - range2_: A tuple (nbins, ymin, ymax) defining the Y-axis binning.
    - title: The title of the histogram.
    - weight: weight

    Returns:
    - A ROOT.RResultPtr<TH2D> object representing the booked 2D histogram.
    """
    # Extract axis parameters
    xbins, xmin, xmax = range1_[:3]
    ybins, ymin, ymax = range2_[:3]

    htitle=title if title is not None else name
    # Create the 2D histogram model
    hist_model = ROOT.ROOT.RDF.TH2DModel(
        name,          # name of the histogram
        htitle,          # Title of the histogram
        xbins, xmin, xmax,  # X-axis bins and range
        ybins, ymin, ymax   # Y-axis bins and range
    )

    # Book the histogram on the RDataFrame
    if weight is not None:
        hist = df.Histo2D(hist_model, var1, var2, weight)
    else:
        hist = df.Histo2D(hist_model, var1, var2)
    #hist = df.Histo2D(model, "pt", "eta", "weight");

    # Return the histogram object
    return hist

def wrap_overflow1d(h):
    """Add underflow to the first bin and overflow to the last bin, then return the updated histogram."""
    hist = h.Clone()
    
    # Get underflow and overflow values
    underflow = hist.GetBinContent(0)  # Bin 0 is the underflow bin
    overflow = hist.GetBinContent(hist.GetNbinsX() + 1)  # Last bin + 1 is the overflow bin

    # Add underflow to the first bin
    first_bin_content = hist.GetBinContent(1)
    hist.SetBinContent(1, first_bin_content + underflow)

    # Add overflow to the last bin
    last_bin_index = hist.GetNbinsX()
    last_bin_content = hist.GetBinContent(last_bin_index)
    hist.SetBinContent(last_bin_index, last_bin_content + overflow)

    return hist


def wrap_overflow2d(histo2D):
        
    nbinx = histo2D.GetNbinsX()
    nbiny = histo2D.GetNbinsY()
    for ix in range(0, nbinx+1):
        # underflow y -> first visible bin
        histo2D.SetBinContent(ix, 1, histo2D.GetBinContent(ix, 0) + histo2D.GetBinContent(ix, 1))
        histo2D.SetBinContent(ix, 0, 0)
        histo2D.SetBinError(ix, 1, (histo2D.GetBinError(ix, 0)**2 + histo2D.GetBinError(ix, 1)**2)**0.5)
        histo2D.SetBinError(ix, 0, 0)
        
        # overflow y -> last visible bin
        histo2D.SetBinContent(ix, nbiny, histo2D.GetBinContent(ix, nbiny) + histo2D.GetBinContent(ix, nbiny+1))
        histo2D.SetBinContent(ix, nbiny+1, 0)
        histo2D.SetBinError(ix, nbiny, (histo2D.GetBinError(ix, nbiny)**2 + histo2D.GetBinError(ix, nbiny+1)**2)**0.5)
        histo2D.SetBinError(ix, nbiny+1, 0)

    # --- absorb underflow/overflow in x ---
    for iy in range(1, nbiny):
        # underflow x -> first visible bin
        histo2D.SetBinContent(1, iy, histo2D.GetBinContent(0, iy) + histo2D.GetBinContent(1, iy))
        histo2D.SetBinContent(0, iy, 0)
        histo2D.SetBinError(1, iy, (histo2D.GetBinError(0, iy)**2 + histo2D.GetBinError(1, iy)**2)**0.5)
        histo2D.SetBinError(0, iy, 0)
        
        # overflow x -> last visible bin
        histo2D.SetBinContent(nbinx, iy, histo2D.GetBinContent(nbinx, iy) + histo2D.GetBinContent(nbinx+1, iy))
        histo2D.SetBinContent(nbinx+1, iy, 0)
        histo2D.SetBinError(nbinx, iy, (histo2D.GetBinError(nbinx, iy)**2 + histo2D.GetBinError(nbinx+1, iy)**2)**0.5)
        histo2D.SetBinError(nbinx+1, iy, 0)

    return histo2D

def normalize_hist(hist):
    """
    Normalize histogram so that the sum of all bin contents equals 1.
    """
    total = hist.Integral()
    if total == 0:
        print("Warning: Histogram total is zero, cannot normalize.")
        return hist

    hist.Scale(1.0 / total)
    return hist