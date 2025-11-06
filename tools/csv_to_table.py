import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

from funcs.common import get_argv, rounding


def csv_to_table(csv_files, output_file, set_roundoff,order_col=None,needed_columns=None):

    def get_max_text_length(df):
        """Return the max length of content in any column or cell"""
        max_len = max([len(str(col)) for col in df.columns])
        for row in df.values:
            for cell in row:
                max_len = max(max_len, len(str(cell)))
        return max_len

    def reorder_by_column(df, column_name, ascending=True):
        """
        Reorders the DataFrame by a specified column.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            column_name (str): Name of the column to sort by.
            ascending (bool): Whether to sort in ascending order. Default is True.
        
        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        
        # Try to convert the column to numeric (in case it’s read as string)
        try:
            df[column_name] = pd.to_numeric(df[column_name])
        except ValueError:
            raise ValueError(f"Column '{column_name}' could not be converted to numeric.")

        return df.sort_values(by=column_name, ascending=ascending).reset_index(drop=True)

    with PdfPages(output_file) as pdf:
        for csv_file in csv_files:
            if not os.path.isfile(csv_file):
                print(f"File not found: {csv_file}")
                continue

            df = pd.read_csv(csv_file)
            if needed_columns is not None:
                missing = [col for col in needed_columns if col not in df.columns]
                if missing:
                    print(f"Warning: Missing columns in {csv_file}: {missing}")
                df = df[[col for col in needed_columns if col in df.columns]]


            df = df.applymap(lambda x: rounding(x, set_roundoff))
            if order_col is not None:
                df = reorder_by_column(df,order_col)
            # Estimate size
            max_text_len = get_max_text_length(df)
            col_width = max(1.2, max_text_len / 10)  # heuristic scaling
            fig_width = col_width * len(df.columns)
            fig_height = max(2, 0.4 * len(df))  # minimum height

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis('tight')
            ax.axis('off')

            ax.set_title(f"Table from {os.path.basename(csv_file)}", fontsize=14, fontweight='bold', pad=20)

            table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.2)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"PDF saved as: {output_file}")

def main(csv_files, output_file, set_roundoff,order_col):
    csv_to_table(csv_files, output_file, set_roundoff,order_col)

if __name__ =="__main__":
    set_roundoff = False
    cols = ['']
    order_col = None
    #order_col = 'act_signif'
    csv_files, output_file, set_roundoff = get_argv("List csv files, output file name, round-off", 3, [(list,str),str,int])
    main(csv_files, output_file, set_roundoff,order_col)
