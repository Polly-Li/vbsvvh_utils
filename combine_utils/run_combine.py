#this is to get combine result so cmsenv is necessary
import subprocess
import tempfile
import os
from configs.paths import base_dir

def run_combine(sig, bkg,obs,print_datacard=True):
    template_path = f"{base_dir}/combine_utils/datacard_template.txt"

    # Read and modify datacard
    with open(template_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("rate"):
            new_lines.append(f"rate      {sig} {bkg}\n")
        elif line.startswith("observation"):
            new_lines.append(f"observation {obs}\n")
        else:
            new_lines.append(line)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_card:
        tmp_card.writelines(new_lines)
        tmp_card_path = tmp_card.name

    # Run combine and capture output
    
    if print_datacard:
        subprocess.run(['cat', tmp_card_path])
    try:
        result = subprocess.run(
            ["combine", tmp_card_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError:
        os.remove(tmp_card_path)
        raise RuntimeError("combine command failed")

    # Clean up temp file
    os.remove(tmp_card_path)

    # Parse expected limits
    limits = {}
    for line in output.splitlines():
        if "Expected  2.5%" in line:
            limits["ll"] = float(line.split("<")[-1].strip())
        elif "Expected 16.0%" in line:
            limits["l"] = float(line.split("<")[-1].strip())
        elif "Expected 50.0%" in line:
            limits["m"] = float(line.split("<")[-1].strip())
        elif "Expected 84.0%" in line:
            limits["u"] = float(line.split("<")[-1].strip())
        elif "Expected 97.5%" in line:
            limits["uu"] = float(line.split("<")[-1].strip())

    return limits  # dict with keys: ll, l, m, u, uu


def run_combine_hybridnew(sig, bkg):
    template_path = f"{base_dir}/combine_utils/datacard_template.txt"

    # Read and modify datacard
    with open(template_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("rate"):
            new_lines.append(f"rate      {sig} {bkg}\n")
        elif line.startswith("observation"):
            new_lines.append(f"observation {sig+bkg}\n")
        else:
            new_lines.append(line)
    print(new_lines)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_card:
        tmp_card.writelines(new_lines)
        tmp_card_path = tmp_card.name

    # Run combine and capture output
    try:
        result = subprocess.run(
            ["combine", "-H", "AsymptoticLimits", "-M", "HybridNew", tmp_card_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True
        )
        output = result.stdout
        print(output)
    except subprocess.CalledProcessError:
        os.remove(tmp_card_path)
        raise RuntimeError("combine command failed")

    # Clean up temp file
    os.remove(tmp_card_path)

    # Parse expected limits
    limit = 0
    for line in output.splitlines():
        if "Limit:" in line:
            limit_str = line.split("<")[-1].strip().split()[0]  # gets '160.095'
            limit = float(limit_str)

    return limit 


if __name__ == "__main__":
    result = run_combine(sig=1.0, bkg=0.5)
    print(result)