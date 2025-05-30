#!/usr/bin/env python3
# import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from training import train_model

def excel_col_letter_to_index(col_letter):
    col_letter = col_letter.upper()
    idx = 0
    for i, char in enumerate(reversed(col_letter)):
        idx += (ord(char) - ord('A') + 1) * (26 ** i)
    return idx - 1

def main():
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = BASE_DIR / "data"
    plot_dir = BASE_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Ask user for Excel filename
    excel_filename = input("Enter the Excel filename (e.g., test.xlsx): ").strip()
    excel_path = data_dir / excel_filename

    # Load the Excel file
    df = pd.read_excel(excel_path, engine="openpyxl")
    df.columns = df.columns.str.strip()

    # Condition Column Selection
    print("\nEnter Condition Column (or 'done', 'all'):")
    condition_columns = []
    condition_values_dict = {}
    col_input = input("  Condition column: ").strip()

    if col_input.lower() == "all":
        start_col = input("  Enter starting condition column: ").strip()
        end_col = input("  Enter ending condition column: ").strip()
        if start_col not in df.columns or end_col not in df.columns:
            raise ValueError("  One or both condition columns not found.")
        start_idx = df.columns.get_loc(start_col)
        end_idx = df.columns.get_loc(end_col)
        if end_idx < start_idx:
            raise ValueError("  Ending condition column comes before start.")
        condition_columns = list(df.columns[start_idx:end_idx + 1])
        for col in condition_columns:
            condition_values_dict[col] = df[col].dropna().unique().tolist()
    else:
        while col_input.lower() != "done":
            if col_input not in df.columns:
                print(f"  Column '{col_input}' not found. Try again.")
            else:
                condition_columns.append(col_input)
                unique_vals = df[col_input].dropna().unique()
                print(f"    Values for '{col_input}': {list(unique_vals)}")
                val_input = input("    Select values (comma-separated) or type 'all': ").strip()
                if val_input.lower() == "all":
                    condition_values_dict[col_input] = list(unique_vals)
                else:
                    condition_values_dict[col_input] = [v.strip() for v in val_input.split(",") if v.strip() in unique_vals]
            col_input = input("  Condition column: ").strip()

    # Species filter
    while True:
        species_col = input("\nEnter Species Column Name (or press Enter to skip): ").strip()

        if species_col == "":
            species_col = None
            selected_species = None
            break

        if species_col in df.columns:
            unique_species = df[species_col].dropna().unique()
            print(f"  Species: {list(unique_species)}")
            species_input = input("  Species to use (or 'done', 'all'): ").strip()
            selected_species = list(unique_species) if species_input.lower() == "all" else [
                s.strip() for s in species_input.split(",") if s.strip() in unique_species
            ]
            break
        else:
            print("  Column not found. Try again or press Enter to skip.")

    rpa_col_input = input("\nEnter column name for Total RPA (or press Enter to skip): ").strip()
    if rpa_col_input:
        try:
            rpa_col = rpa_col_input if rpa_col_input in df.columns else df.columns[excel_col_letter_to_index(rpa_col_input)]
            min_rpa = float(input("  Enter minimum Total RPA value to include: ").strip())
        except:
            raise ValueError("Invalid RPA column or minimum value.")
    else:
        rpa_col = None
        min_rpa = None

    mask = pd.Series(True, index=df.index)
    for col in condition_columns:
        mask &= df[col].isin(condition_values_dict[col])
    if species_col and selected_species:
        mask &= df[species_col].isin(selected_species)
    if rpa_col and min_rpa is not None:
        mask &= df[rpa_col].astype(float) >= min_rpa
        print(f"Filtered rows with {rpa_col} ≥ {min_rpa}. Remaining rows: {mask.sum()}")

    df = df[mask].reset_index(drop=True)

    # Column Selection
    while True:
        print("\nEnter gene column names (e.g. 'Complement C3', 'Clusterin').")
        choice = input("  Use a range of gene columns? (Enter Y/N/ALL): ").strip().upper()

        if choice == "Y":
            start_col = input("  Enter starting gene column: ").strip()
            end_col = input("  Enter ending gene column: ").strip()
            try:
                start_gene_idx = df.columns.get_loc(start_col) if start_col in df.columns else excel_col_letter_to_index(start_col)
                end_gene_idx = df.columns.get_loc(end_col) if end_col in df.columns else excel_col_letter_to_index(end_col)
                if not (0 <= start_gene_idx <= end_gene_idx < len(df.columns)):
                    raise ValueError("Invalid column range.")
                gene_names = list(df.columns[start_gene_idx:end_gene_idx + 1])
                # Keep a copy of all gene names for later filtering

                break
            except Exception as e:
                print(f"  Invalid input: {e}")

        elif choice == "ALL":
            start_col = input("  Enter starting gene column: ").strip()
            try:
                start_gene_idx = df.columns.get_loc(start_col) if start_col in df.columns else excel_col_letter_to_index(start_col)
                if not (0 <= start_gene_idx < len(df.columns)):
                    raise ValueError("Invalid starting column.")
                gene_names = list(df.columns[start_gene_idx:])
                break
            except Exception as e:
                print(f"  Invalid input: {e}")

        elif choice == "N":
            gene_names = []
            while True:
                gene_input = input("  Column name (or 'done'): ").strip()
                if gene_input.lower() == "done":
                    break
                if gene_input in df.columns:
                    gene_names.append(gene_input)
                else:
                    try:
                        idx = excel_col_letter_to_index(gene_input)
                        if 0 <= idx < len(df.columns):
                            gene_names.append(df.columns[idx])
                        else:
                            print(f"  Column letter '{gene_input}' out of bounds.")
                    except:
                        print(f"  Column '{gene_input}' not found. Try again.")
            break

        else:
            print("  Invalid input. Please enter 'Y', 'N', or 'ALL'.")

    X_raw = df[gene_names].values.astype(np.float32)
    print(f"\n Selected {len(gene_names)} gene columns. Matrix size: {X_raw.shape}")

    df_selected = df[condition_columns].reset_index(drop=True).join(pd.DataFrame(X_raw, columns=gene_names))
    df_selected.to_csv(data_dir / "input.csv", index=False)

    # Encode conditions and split data 
    encoders = {
        col: {v: [1 if i == j else 0 for j, _ in enumerate(vals)]
            for i, v in enumerate(vals)}
        for col, vals in ((col, df[col].unique().tolist()) for col in condition_columns)
    }
    C_df = pd.get_dummies(df[condition_columns], drop_first=False)
    cond_names = C_df.columns.tolist()             # now matches C_df.shape[1]
    C = C_df.values.astype(np.float32)             # e.g. (110, 40)
    # cond_dim = C.shape[1]
    gene_names_all = gene_names.copy()

    Xtr_raw, Xv_raw, Ctr, Cv = train_test_split(X_raw, C, test_size=0.2, random_state=42)

    # Normalize gene expression
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xv = scaler.transform(Xv_raw)
    latent_dim=256
    ks_boost = 2.0
    # Call training
    enc, dec, gene_names = train_model(Xtr, Xv, Ctr, Cv, cond_names, gene_names, gene_names_all, scaler, data_dir, plot_dir, latent_dim=latent_dim,ks_boost=ks_boost)

if __name__ == "__main__":
    main()
