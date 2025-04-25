# relabel_app.py

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np # Import numpy for NaN comparison

# --- UI Setup ---
st.set_page_config(page_title="Label Correction Tool", layout="centered")

# Define file paths
INPUT_CSV_PATH = Path("../reports/label_review/suspected_label_errors.csv")
OUTPUT_CSV_PATH = Path("../reports/label_review/corrected_labels.csv")

# --- Data Loading and State Initialization ---
def load_data():
    """Loads data, preferring the output/state file if it exists."""
    if OUTPUT_CSV_PATH.exists():
        df = pd.read_csv(OUTPUT_CSV_PATH)
        # Ensure Corrected column exists, even if loading old output file
        if "Corrected" not in df.columns:
             df["Corrected"] = pd.NA
        st.info(f"Resuming session from: `{OUTPUT_CSV_PATH}`")
    else:
        if not INPUT_CSV_PATH.exists():
             st.error(f"Input file not found: {INPUT_CSV_PATH}")
             st.stop()
        df = pd.read_csv(INPUT_CSV_PATH)
        df["Corrected"] = pd.NA # Add Corrected column for the first run
        st.info(f"Starting new session from: `{INPUT_CSV_PATH}`")
    # Ensure Corrected column has object dtype to store strings or NA
    df['Corrected'] = df['Corrected'].astype(object)
    return df

def find_start_index(df):
    """Finds the index of the first row without a correction."""
    # Get boolean series where True means 'Corrected' is NA
    is_na_series = df['Corrected'].isna()
    # Find the index of the first True value
    first_na_index = is_na_series.idxmax()

    # Check if the value at the found index is actually NA
    # idxmax returns the first index (0) if no True exists
    if is_na_series.any() and pd.isna(df.loc[first_na_index, 'Corrected']):
        return first_na_index
    elif not is_na_series.any():
         # If no NA values exist at all, all are corrected
         return len(df) # Start at the end if all are done
    else:
         # This case might happen if idxmax returned 0 but df.loc[0, 'Corrected'] is not NA
         # (meaning only the first item was corrected, but others might be NA)
         # Fallback to a simple search, though the above logic should cover it.
         for idx, value in df['Corrected'].items():
             if pd.isna(value):
                 return idx
         return len(df) # Should not be reached if any NA exists


# Load data at the start
df = load_data()

# --- Initialize session state ONCE per session ---
if 'initialized' not in st.session_state:
    st.session_state.row_idx = find_start_index(df)
    st.session_state.relabel_input = ""
    st.session_state.df = df # Store df in session state to make it mutable across reruns
    st.session_state.initialized = True
else:
    # On subsequent reruns, use the DataFrame from session state
    df = st.session_state.df


# Categories (sorted and fixed order) - Derive from loaded data
all_found_categories = df["True"].dropna().unique().tolist() + df["Predicted"].dropna().unique().tolist()
all_categories = sorted(list(set(cat for cat in all_found_categories if pd.notna(cat))))
category_keys = {str(i+1): cat for i, cat in enumerate(all_categories)}
num_categories = len(category_keys)
# Create a reverse mapping for easy lookup
category_to_key = {v: k for k, v in category_keys.items()}

# --- UI Setup ---
st.title("📝 Review and Relabel Suspected Label Errors")
st.caption(f"Press 1–{num_categories} to relabel, then press Enter. Progress is auto-saved to `{OUTPUT_CSV_PATH.name}`.")

# --- Core Logic Functions ---
def save_progress(dataframe):
    """Saves the entire DataFrame to the output file."""
    try:
        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(OUTPUT_CSV_PATH, index=False)
        # Optionally add a success message here if needed, but might be too noisy
    except Exception as e:
        st.error(f"Error saving progress: {e}")

def next_row():
    current_idx = st.session_state.row_idx
    # Use df from session state for length check
    st.session_state.row_idx = min(current_idx + 1, len(st.session_state.df))

def prev_row():
    current_idx = st.session_state.row_idx
    st.session_state.row_idx = max(0, current_idx - 1)

def process_relabel():
    """Callback to process input, update DataFrame, save, and advance."""
    pressed = st.session_state.relabel_input
    current_idx = st.session_state.row_idx
    session_df = st.session_state.df # Use DataFrame from session state

    if current_idx < len(session_df) and pressed in category_keys:
        corrected_label = category_keys[pressed]
        # Update the DataFrame IN SESSION STATE
        session_df.loc[current_idx, "Corrected"] = corrected_label
        st.session_state.relabel_input = "" # Clear input immediately

        # Auto-save progress using the updated session DataFrame
        save_progress(session_df)

        next_row()
        # Rerun is implicitly triggered by state change


# --- Main Display Loop ---
i = st.session_state.row_idx
current_df = st.session_state.df # Use df from session state for display

if i >= len(current_df):
    st.success("✅ You've reviewed all flagged examples!")
    st.info(f"All corrections saved to `{OUTPUT_CSV_PATH}`")
else:
    row = current_df.iloc[i]
    # Progress bar calculation needs to handle df length correctly
    progress_value = min((i + 1) / len(current_df), 1.0) if len(current_df) > 0 else 0
    st.progress(progress_value, text=f"Item {i+1} of {len(current_df)}")

    st.markdown(f"**Question:** {row.get('Question', 'N/A')}") # Use .get for safety
    st.markdown(f"**Answer:** {row.get('Answer', 'N/A')}")

    predicted_label = row.get('Predicted', 'N/A')
    predicted_key = category_to_key.get(predicted_label, '?')
    st.markdown(f"**Predicted:** `{predicted_label} [{predicted_key}]`")

    original_label = row.get('True', 'N/A')
    original_key = category_to_key.get(original_label, '?')
    st.markdown(f"**Original Label:** `{original_label} [{original_key}]`")

    # Display corrected label if it exists
    corrected_val = row.get("Corrected")
    if pd.notna(corrected_val):
        st.markdown(f"**Corrected Label:** `{corrected_val}`")
    else:
        st.markdown("**Corrected Label:** _Not yet corrected_")

    # Handle potential missing Confidence
    confidence = row.get('Confidence', None)
    if confidence is not None:
        st.markdown(f"**Model Confidence:** {confidence:.4f}")
    else:
        st.markdown("**Model Confidence:** _N/A_")


    # Display category keys
    st.markdown("### Relabel options:")
    # Responsive columns: Adjust based on number of categories
    num_cols = min(len(category_keys), 5) # Max 5 columns
    cols = st.columns(num_cols)
    col_idx = 0
    for key, cat in category_keys.items():
        cols[col_idx % num_cols].markdown(f"**[{key}]** {cat}")
        col_idx += 1

    # Add some space
    st.markdown("---")

    # Relabel input - Uses session state key and on_change callback
    st.text_input(
        f"Enter number (1–{num_categories}) and press Enter to correct:",
        key="relabel_input",
        on_change=process_relabel,
        value=st.session_state.relabel_input, # Ensure input reflects state
        placeholder="Type number and press Enter..."
    )

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 5]) # Spacing
    with col1:
        st.button("⬅️ Back", on_click=prev_row, disabled=(i <= 0), use_container_width=True)
    with col2:
        st.button("➡️ Next", on_click=next_row, disabled=(i >= len(current_df) - 1), use_container_width=True)


# --- Export / Final Save (Optional) ---
st.divider()
# Button is less critical now due to auto-save, but can be kept for clarity
if st.button("💾 Verify Save (Auto-Saved Anyway)"):
    # Re-save the latest state from session state
    save_progress(st.session_state.df)
    st.success(f"Current progress verified and saved to `{OUTPUT_CSV_PATH}`")

# Display path for clarity
st.caption(f"Input: `{INPUT_CSV_PATH}` | Output/State: `{OUTPUT_CSV_PATH}`")
