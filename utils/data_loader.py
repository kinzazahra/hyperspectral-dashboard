import scipy.io as sio
import streamlit as st

@st.cache_data
def load_mat_file(file):
    mat = sio.loadmat(file)
    return mat

def extract_data(mat_dict):
    # Remove metadata keys
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    return mat_dict[keys[0]]