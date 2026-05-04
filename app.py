import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.io import wavfile
import io
import h5py
from qutip import mesolve, Qobj
from scipy.interpolate import interp1d
import torch

# --- 1. THE UI SETUP ---
st.set_page_config(page_title="Quantum Harmonium", layout="wide")
st.markdown("# 🎶 Quantum Harmonium\n  Two electrons in a Harmonic Trap")
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

# --- 3. PYTORCH OPTIMAL CONTROL ENGINE ---
def optimize_control_pulse(H_np, mu_np, psi0_np, target_psi_np, t_max, n_steps=100, epochs=50, lr=0.05):
    """
    Uses PyTorch Autograd to find the optimal laser pulse u(t) 
    that drives the system from psi0 to target_psi.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = t_max / n_steps
    
    H = torch.tensor(H_np, dtype=torch.complex64, device=device)
    mu = torch.tensor(mu_np, dtype=torch.complex64, device=device)
    psi0 = torch.tensor(psi0_np, dtype=torch.complex64, device=device).unsqueeze(1)
    target_psi = torch.tensor(target_psi_np, dtype=torch.complex64, device=device).unsqueeze(1)
    
    u = torch.zeros(n_steps, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([u], lr=lr)
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        psi = psi0
        
        for i in range(n_steps):
            H_tot = H + u[i] * mu
            U = torch.matrix_exp(-1j * dt * H_tot)
            psi = torch.matmul(U, psi)
            
        overlap = torch.abs(torch.vdot(target_psi.squeeze(), psi.squeeze()))**2
        loss = 1.0 - overlap
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        
    # NEW: Return the final psi vector as a numpy array along with the pulse data
    return u.detach().cpu().numpy(), loss_history, np.linspace(0, t_max, n_steps), psi.detach().cpu().numpy().flatten()
    
# --- 4. LOAD REAL MATRICES ---
st.sidebar.header("System Parameters")
try:
    with h5py.File('main.h5', 'r') as h_file: 
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
tab1, tab2, tab3 = st.tabs(["🎛️ The Synthesizer (QuTiP)", "🎚️ The Paddle Board", "🧠 PyTorch Optimal Control"])

if H_matrix is not None and mu_matrix is not None:
    energies, states = np.linalg.eigh(H_matrix)
    n_states = len(energies)

# ==========================================
# TAB 1: THE SYNTHESIZER
# ==========================================
with tab1:
    if H_matrix is not None and mu_matrix is not None:
        st.markdown("### Dynamics Controls")
        col_1, col_2, col_3 = st.columns(3)
        
        with col_1:
            initial_state_idx = st.number_input("Initial State |ψ₀⟩", min_value=0, max_value=n_states-1, value=0, key="t1_init")
        with col_2:
            drive_freq = st.number_input("Drive Frequency (ω /atomic time)", value=1.0, step=0.1)
        with col_3:
            drive_amp = st.number_input("Drive Amplitude (A au)", value=0.5, step=0.1)

        st.markdown("### Audio Controls")
        col_a, col_b = st.columns(2)
        with col_a:
            t_max_input = st.number_input("Signal Duration (atomic time)", min_value=0.1, max_value=600.0, value=60.0, step=0.5)
        with col_b:
            sample_rate_input = st.selectbox("Sample Rate (Resolution)", options=[100, 220, 441, 960], index=0)

        if st.button("Simulate Dynamics", type="primary"):
            with st.spinner("Solving Master Equation..."):
                psi0_vec = states[:, initial_state_idx]
                psi0 = Qobj(psi0_vec)
                Obs = [Qobj(mu_matrix)]
                Ops = []
                
                def drive(t, args):
                    return args['A'] * np.cos(args['w'] * t)
                
                args = {'A': drive_amp, 'w': drive_freq}

                t, audio_out = compute_quantum_signal(
                    H_matrix, mu_matrix, Ops=Ops, Obs=Obs, psi0=psi0, 
                    drive_func=drive, args=args, t_max=t_max_input, sample_rate=sample_rate_input
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=audio_out, mode='lines', name='⟨μ⟩ Expectation', line=dict(color='teal', width=1)))
                fig.update_layout(title="Time-Domain Radiation (Dipole Expectation Value)", xaxis_title="Time (au)", yaxis_title="⟨μ⟩", template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
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
        mu_eigen = states.T.conj() @ mu_matrix @ states
        global_max_projection = np.max(np.abs(mu_eigen))
        
        ket_idx = st.number_input("Select Ket |j⟩ (Index) for Scattering", min_value=0, max_value=n_states-1, value=0)
        ket_vector = states[:, ket_idx]
            
        bra_vector = mu_matrix @ ket_vector
            
        st.markdown(f"**Transition Spectrum for Initial State |{ket_idx}⟩:**")
        projections = np.abs(states.T.conj() @ bra_vector)
        
        fig_bar = go.Figure(data=[go.Bar(x=[f"|{i}⟩" for i in range(n_states)], y=projections.flatten(), marker_color='orange')])
        fig_bar.update_layout(yaxis=dict(range=[0, global_max_projection * 1.1]), xaxis=dict(type='category'), xaxis_title="Eigenstates", yaxis_title="Amplitude Projection", template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 3: PYTORCH OPTIMAL CONTROL
# ==========================================
with tab3:
    st.markdown("### 🧠 Quantum Pulse Optimization (PyTorch)")
    st.markdown("Let an AI sculpt the exact laser pulse required to force the electrons into a specific target state.")
    
    if H_matrix is not None and mu_matrix is not None:
        # Added a 4th column for the duration input
        col_start, col_target, col_epochs, col_duration = st.columns(4)
        
        with col_start:
            qoc_init_idx = st.number_input("Start State |ψ₀⟩", min_value=0, max_value=n_states-1, value=0, key="qoc_init")
        with col_target:
            qoc_target_idx = st.number_input("Target State |ψ_target⟩", min_value=0, max_value=n_states-1, value=1, key="qoc_targ")
        with col_epochs:
            qoc_epochs = st.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10)
        with col_duration:
            # NEW: User defines how long the AI has to perform the transition
            qoc_duration = st.number_input("Pulse Duration", min_value=1.0, max_value=200.0, value=10.0, step=1.0)
            
        if st.button("Sculpt Laser Pulse (Train)", type="primary"):
            with st.spinner(f"PyTorch is sculpting a {qoc_duration} atomic time unit pulse..."):
                psi0_vec = states[:, qoc_init_idx]
                target_vec = states[:, qoc_target_idx]
                
                # Pass the new qoc_duration to the optimizer
                optimal_pulse, loss_hist, t_pulse, final_psi = optimize_control_pulse(
                    H_matrix, mu_matrix, psi0_vec, target_vec, 
                    t_max=qoc_duration, n_steps=200, epochs=qoc_epochs, lr=0.1
                )
                
                # Calculate State Occupations
                initial_pops = np.abs(states.T.conj() @ psi0_vec)**2
                final_pops = np.abs(states.T.conj() @ final_psi)**2
                
                # Plot the Occupations
                fig_pops = go.Figure(data=[
                    go.Bar(name='Initial State', x=[f"|{i}⟩" for i in range(n_states)], y=initial_pops, marker_color='teal'),
                    go.Bar(name='Final State', x=[f"|{i}⟩" for i in range(n_states)], y=final_pops, marker_color='magenta')
                ])
                fig_pops.update_layout(
                    title="State Occupation (Population Dynamics)",
                    xaxis_title="Eigenstates", 
                    yaxis_title="Population Probability |⟨j|ψ⟩|²",
                    barmode='group', 
                    template="plotly_dark", 
                    height=300, 
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_pops, use_container_width=True)

                # Plot the Training Loss
                fig_loss = go.Figure(data=[go.Scatter(y=loss_hist, mode='lines', line=dict(color='magenta'))])
                fig_loss.update_layout(title="Pulse Training Infidelity (Lower is closer to target state)", xaxis_title="Epoch", yaxis_title="1 - Fidelity", template="plotly_dark", height=250, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Plot the AI-Generated Pulse u(t)
                fig_pulse = go.Figure(data=[go.Scatter(x=t_pulse, y=optimal_pulse, mode='lines', line=dict(color='yellow'))])
                fig_pulse.update_layout(title="The Optimized Driving Pulse u(t)", xaxis_title="Atomic Time", yaxis_title="Amplitude", template="plotly_dark", height=250, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_pulse, use_container_width=True)
                
                # --- INTEGRATE BACK INTO QUTIP FOR AUDIO ---
                st.success("Pulse Optimized! Simulating final audio output using QuTiP...")
                
                pulse_interp = interp1d(t_pulse, optimal_pulse, bounds_error=False, fill_value=0.0)
                
                def drive_opt(t, args):
                    return float(pulse_interp(t))
                
                # Ensure the QuTiP verifier also uses the new duration
                t_audio, audio_out = compute_quantum_signal(
                    H_matrix, mu_matrix, Ops=[], Obs=[Qobj(mu_matrix)], psi0=Qobj(psi0_vec), 
                    drive_func=drive_opt, args={}, t_max=qoc_duration, sample_rate=441
                )
                
                
                if False:
                   audio_data = np.int16(audio_out * 32767)
                   virtual_file = io.BytesIO()
                   wavfile.write(virtual_file, 441, audio_data)
                   st.audio(virtual_file, format="audio/wav")