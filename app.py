import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.io import wavfile
import io
import h5py
from qutip import mesolve, Qobj
#from scipy.signal import resample


# --- 1. THE UI SETUP ---
st.set_page_config(page_title="Quantum Harmonium", layout="wide")
st.markdown("# 🎶 Quantum Harmonium\n   two electrons in a Harmonic Trap")
st.markdown("This app uses QuTiP to solve the time-dependent quantum dynamics in spectral basis. Drive the harmonium with a laser and listen to the dipole radiation.")

# --- 2. THE CORE MATH ENGINE ---
def compute_quantum_signal(H, mu, Ops, Obs, psi0, drive_func, args, t_max=10.0, sample_rate=44100):
    tlist = np.linspace(0, t_max, int(sample_rate * t_max))
    
    # Format the time-dependent Hamiltonian for QuTiP
    H_td = [Qobj(H), [Qobj(mu), drive_func]] 
    
    # Solve the Master Equation
    result = mesolve(H_td, psi0, tlist, c_ops=Ops, e_ops=Obs, args=args)
    
    # Extract the expectation value of our observable (the dipole)
    signal = result.expect[0]
    
    # Normalize for audio playback (-1.0 to 1.0)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
        
    return tlist, signal

# --- 3. LOAD REAL MATRICES ---
st.sidebar.header("System Parameters")
try:
    with h5py.File('main.h5', 'r') as h_file: # Note: Adjust to main5.h5 or main.h5 based on your actual file
        H_matrix = np.array(h_file['main_hamiltonian'])
    with h5py.File('x.h5', 'r') as x_file:
        mu_matrix = np.array(x_file['dipole-x']) * 1j
        
    st.sidebar.success("✅ Matrices Loaded Successfully")
    st.sidebar.write(f"Hamiltonian: `{H_matrix.shape}`")
    st.sidebar.write(f"Dipole: `{mu_matrix.shape}`")
except Exception as e:
    st.sidebar.error(f"Error reading files: {e}")
    H_matrix, mu_matrix = None, None

# --- CREATE THE TABS ---
tab1, tab2 = st.tabs(["🎛️ The Synthesizer (QuTiP)", "🎚️ The Paddle Board"])

# ==========================================
# TAB 1: THE SYNTHESIZER
# ==========================================
with tab1:
    if H_matrix is not None and mu_matrix is not None:
        # Pre-diagonalize to get our eigenbasis for the initial state selector
        energies, states = np.linalg.eigh(H_matrix)
        n_states = len(energies)

        st.markdown("### Dynamics Controls")
        col_1, col_2, col_3 = st.columns(3)
        
        with col_1:
            initial_state_idx = st.number_input("Initial State |ψ₀⟩ (Index)", min_value=0, max_value=n_states-1, value=0)
        with col_2:
            drive_freq = st.number_input("Drive Frequency (ω)", value=1.0, step=0.1)
        with col_3:
            drive_amp = st.number_input("Drive Amplitude (A)", value=0.5, step=0.1)

        st.markdown("### Audio Controls")
        col_a, col_b = st.columns(2)
        with col_a:
            t_max_input = st.number_input("Signal Duration (Seconds)", min_value=0.1, max_value=600.0, value=60.0, step=0.5)
        with col_b:
            sample_rate_input = st.selectbox("Sample Rate (Resolution)", options=[100, 220, 441, 960], index=0)

        if st.button("Simulate Dynamics", type="primary"):
            with st.spinner("Solving Master Equation..."):
                
                # 1. Define Initial State (psi0)
                psi0_vec = states[:, initial_state_idx]
                psi0 = Qobj(psi0_vec)
                
                # 2. Define Observable (Measure the dipole expectation value)
                Obs = [Qobj(mu_matrix)]
                
                # 3. Define Collapse Operators (Empty list = Unitary evolution)
                Ops = []
                
                # 4. Define the Driver Function and Arguments
                def drive(t, args):
                    return args['A'] * np.cos(args['w'] * t)
                
                args = {'A': drive_amp, 'w': drive_freq}

                # 5. Execute
                t, audio_out = compute_quantum_signal(
                    H_matrix, mu_matrix, 
                    Ops=Ops, Obs=Obs, psi0=psi0, 
                    drive_func=drive, args=args, 
                    t_max=t_max_input, sample_rate=sample_rate_input
                )
                
                
                # Plotly Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=audio_out, mode='lines', name='⟨μ⟩ Expectation', line=dict(color='teal', width=1)))
                fig.update_layout(
                    title="Time-Domain Radiation (Dipole Expectation Value)",
                    xaxis_title="Time (t)", yaxis_title="⟨μ⟩",
                    template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Audio Engine
                # Resample calculates the smooth curve and generates a matched time array
                #new_length = len(audio_out)
                #audio_out_100x, t_100x = resample(audio_out, new_length, t)
                
                audio_data = np.int16(audio_out * 32767)
                virtual_file = io.BytesIO()
                wavfile.write(virtual_file, sample_rate_input, audio_data)
                
                st.audio(virtual_file, format="audio/wav")

# ==========================================
# TAB 2: THE PADDLE BOARD
# ==========================================
with tab2:
    st.markdown("### 🎚️ The Paddle Board")
    st.markdown("`x-dipole * ket = bra`")

    if H_matrix is not None and mu_matrix is not None:
        energies, states = np.linalg.eigh(H_matrix)
        n_states = len(energies)
        
        mu_eigen = states.T.conj() @ mu_matrix @ states
        global_max_projection = np.max(np.abs(mu_eigen))
        
        ket_idx = st.number_input("Select Ket |j⟩ (Index) for Scattering", min_value=0, max_value=n_states-1, value=0)
        ket_vector = states[:, ket_idx]
            
        bra_vector = mu_matrix @ ket_vector
            
        st.markdown(f"**Transition Spectrum for Initial State |{ket_idx}⟩:**")
        projections = np.abs(states.T.conj() @ bra_vector)
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=[f"|{i}⟩" for i in range(n_states)], 
                y=projections.flatten(),
                marker_color='orange'
            )
        ])
        
        fig_bar.update_layout(
            yaxis=dict(range=[0, global_max_projection * 1.1]),
            xaxis=dict(type='category'), 
            xaxis_title="Eigenstates",
            yaxis_title="Amplitude Projection",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)