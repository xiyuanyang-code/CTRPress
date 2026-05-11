#!/usr/bin/env python3
"""Generate LaTeX tables from CSV data."""

import pandas as pd
import os

def read_csv_data(dataset, ratio):
    """Read CSV data for a specific dataset and ratio."""
    filename = f"tables/{dataset}_pythia_ratio{ratio}.csv"
    df = pd.read_csv(filename)
    return df

def format_number(value, precision=2):
    """Format number with specified precision."""
    if value >= 1000:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"

def calculate_percentage(new_val, base_val):
    """Calculate percentage difference."""
    if base_val == 0:
        return 0.0
    return (new_val - base_val) / base_val * 100

def format_percentage(pct, higher_is_better=False):
    """Format percentage with color.

    Args:
        pct: percentage difference
        higher_is_better: if True, positive is green (better), negative is red (worse)
                         if False (default), negative is green (better), positive is red (worse)
    """
    if abs(pct) < 0.01:
        return r"{\tiny\color{gray}(0.0\%)}"
    elif higher_is_better:
        # For metrics like throughput where higher is better
        if pct > 0:
            return r"{\tiny\color{green!60!black}(+" + f"{abs(pct):.1f}" + r"\%)}"
        else:
            return r"{\tiny\color{red!70!black}(-" + f"{abs(pct):.1f}" + r"\%)}"
    else:
        # For metrics like PPL, time where lower is better
        if pct > 0:
            return r"{\tiny\color{red!70!black}(+" + f"{abs(pct):.1f}" + r"\%)}"
        else:
            return r"{\tiny\color{green!60!black}(-" + f"{abs(pct):.1f}" + r"\%)}"

def generate_table(dataset):
    """Generate LaTeX table for a dataset."""
    ratios = [0.3, 0.5, 0.7]
    methods = ['random', 'streaming_llm', 'snapkv', 'lagkv', 'keydiff']

    # Read base data (no compression)
    base_df = read_csv_data(dataset, ratios[0])
    base_row = base_df[base_df['method'] == 'no_compress'].iloc[0]

    # Find best PPL for each ratio
    best_ppl = {}
    for ratio in ratios:
        df = read_csv_data(dataset, ratio)
        method_df = df[df['method'].isin(methods)]
        best_ppl[ratio] = method_df['ppl'].min()

    # Generate table
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \footnotesize")
    lines.append(f"    \\caption{{Main results on {dataset.capitalize()} dataset with Pythia model. Best scores are in \\textbf{{bold}}.}}")
    lines.append(r"    \vspace{-0.2cm}")
    lines.append(f"    \\label{{tab:{dataset}}}")
    lines.append(r"    \setlength{\tabcolsep}{2.5pt}")
    lines.append(r"    \renewcommand{\arraystretch}{1.1}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{l|cccccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Method} & \textbf{PPL} & \textbf{Front} & \textbf{Mid} & \textbf{Back} & \textbf{PreT} & \textbf{TTFT} & \textbf{T/Tok} & \textbf{GenT} & \textbf{TP} & \textbf{Mem} \\")
    lines.append(r"    \midrule")

    # No compression row
    lines.append(r"    \multicolumn{11}{c}{\cellcolor{yellow!12}\textbf{No Compression}} \\")
    lines.append(r"    \rowcolor{gray!8}")
    lines.append(f"    no\\_compress & {format_number(base_row['ppl'])} & {format_number(base_row['front_ppl'])} & {format_number(base_row['middle_ppl'])} & {format_number(base_row['back_ppl'])} & {format_number(base_row['prefilling_time'], 3)} & {format_number(base_row['ttft'], 3)} & {format_number(base_row['time_per_token'], 3)} & {format_number(base_row['generation_time'], 3)} & {format_number(base_row['throughput'], 2)} & {format_number(base_row['peak_memory_usage'], 2)} \\\\")

    # Compression ratio sections
    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        lines.append(r"    \midrule")
        lines.append(f"    \\multicolumn{{11}}{{c}}{{\\cellcolor{{orange!12}}\\textbf{{Compression Ratio {ratio_pct}\\%}}}} \\\\")

        df = read_csv_data(dataset, ratio)

        for method in methods:
            row = df[df['method'] == method].iloc[0]
            is_best = row['ppl'] == best_ppl[ratio]

            # Method name
            method_name = method.replace('_', r'\_')
            ppl_str = f"\\textbf{{{format_number(row['ppl'])}}}" if is_best else format_number(row['ppl'])

            # Calculate percentages for all columns
            ppl_pct = calculate_percentage(row['ppl'], base_row['ppl'])
            front_pct = calculate_percentage(row['front_ppl'], base_row['front_ppl'])
            mid_pct = calculate_percentage(row['middle_ppl'], base_row['middle_ppl'])
            back_pct = calculate_percentage(row['back_ppl'], base_row['back_ppl'])
            pret_pct = calculate_percentage(row['prefilling_time'], base_row['prefilling_time'])
            ttft_pct = calculate_percentage(row['ttft'], base_row['ttft'])
            tptok_pct = calculate_percentage(row['time_per_token'], base_row['time_per_token'])
            gent_pct = calculate_percentage(row['generation_time'], base_row['generation_time'])
            tp_pct = calculate_percentage(row['throughput'], base_row['throughput'])
            mem_pct = calculate_percentage(row['peak_memory_usage'], base_row['peak_memory_usage'])

            # Data row
            lines.append(f"    {method_name} & {ppl_str} & {format_number(row['front_ppl'])} & {format_number(row['middle_ppl'])} & {format_number(row['back_ppl'])} & {format_number(row['prefilling_time'], 3)} & {format_number(row['ttft'], 3)} & {format_number(row['time_per_token'], 3)} & {format_number(row['generation_time'], 3)} & {format_number(row['throughput'], 2)} & {format_number(row['peak_memory_usage'], 2)} \\\\")

            # Percentage row (TP uses higher_is_better=True)
            lines.append(f"     & {format_percentage(ppl_pct)} & {format_percentage(front_pct)} & {format_percentage(mid_pct)} & {format_percentage(back_pct)} & {format_percentage(pret_pct)} & {format_percentage(ttft_pct)} & {format_percentage(tptok_pct)} & {format_percentage(gent_pct)} & {format_percentage(tp_pct, higher_is_better=True)} & {format_percentage(mem_pct)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"    \vspace{0.1cm}")
    lines.append(r"    \footnotesize{PreT: Prefilling Time (s), TTFT: Time to First Token (s), T/Tok: Time per Token (ms), GenT: Generation Time (s), TP: Throughput (tokens/s), Mem: Peak Memory (GB)}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)

def main():
    """Main function."""
    datasets = ['nolima', 'pg19', 'wikitext']

    output = []
    output.append("% Main Table Results")
    output.append("")

    for dataset in datasets:
        output.append(f"% {'='*60}")
        output.append(f"% Table: {dataset.capitalize()} Dataset")
        output.append(f"% {'='*60}")
        output.append("")
        output.append(generate_table(dataset))
        output.append("")

    # Write to file
    with open('main_table.tex', 'w') as f:
        f.write('\n'.join(output))

    print("Generated main_table.tex")

if __name__ == "__main__":
    main()
