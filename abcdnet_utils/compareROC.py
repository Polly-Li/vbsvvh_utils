# to plot and compare ROC
import ROOT

def plot_two_graphs(file1, graph1_name, file2, graph2_name,legends, output_name="ROC.png"):
    # Open the ROOT files
    f1 = ROOT.TFile.Open(file1)
    f2 = ROOT.TFile.Open(file2)
    if not f1 or f1.IsZombie():
        raise FileNotFoundError(f"Cannot open file: {file1}")
    if not f2 or f2.IsZombie():
        raise FileNotFoundError(f"Cannot open file: {file2}")

    # Retrieve the graphs
    g1 = f1.Get(graph1_name)
    g2 = f2.Get(graph2_name)
    if not g1:
        raise KeyError(f"Graph {graph1_name} not found in {file1}")
    if not g2:
        raise KeyError(f"Graph {graph2_name} not found in {file2}")

    # Style setup
    g1.SetLineColor(ROOT.kRed)
    g1.SetLineWidth(2)
    g1.SetMarkerColor(ROOT.kRed)
    g1.SetMarkerStyle(20)

    g2.SetLineColor(ROOT.kBlue)
    g2.SetLineWidth(2)
    g2.SetMarkerColor(ROOT.kBlue)
    g2.SetMarkerStyle(21)

    # Create canvas
    c = ROOT.TCanvas("c", "ROC comparison", 800, 600)
    c.SetGrid()

    # Draw first graph
    g1.SetTitle("ROC Comparison;False Positive Rate;True Positive Rate")
    g1.Draw("ALP")
    g2.Draw("LP SAME")

    # Add legend
    legend = ROOT.TLegend(0.15, 0.15, 0.45, 0.3)
    legend.AddEntry(g1, legends[0], "lp")
    legend.AddEntry(g2, legends[1], "lp")
    legend.Draw()

    # Save
    c.SaveAs(output_name)
    print(f"Saved plot as {output_name}")

    # Clean up
    f1.Close()
    f2.Close()


if __name__ == "__main__":
    # Example usage: adjust file and graph names
    file1 = "/home/users/pyli/outputs/ABCD_trained/c2v1p5_noscores/lambda9/result_plots2.root"
    graph1_name = "ROC/ROC1"
    file2 = "/home/users/pyli/outputs/ABCD_trained/c2v1p5_noscores/lambda10/result_plots2.root"
    graph2_name = "ROC/ROC2"
    legends = ['with scores','wo scores']
    plot_two_graphs(file1, graph1_name, file2, graph1_name,legends, "ROC1.png")
    plot_two_graphs(file1, graph2_name, file2, graph2_name,legends, "ROC2.png")