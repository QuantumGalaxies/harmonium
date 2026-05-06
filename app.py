import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.io import wavfile
import io
import h5py
import os
from qutip import mesolve, Qobj
from scipy.interpolate import interp1d
import torch

# --- 1. THE UI SETUP ---
st.set_page_config(page_title="Quantum Harmonium", layout="wide")
st.markdown("# 🎶 Quantum Harmonium\n  N-electrons in a Harmonic Trap")
st.markdown("Drive the harmonium with a laser, analyze 3D dynamics, and compare correlation effects across different N.")

# --- 2. CORE MATH ENGINES ---
def compute_quantum_signal(H, mu, Ops, Obs, psi0, drive_func, args, t_max=10.0, sample_rate=44100):
    tlist = np.linspace(0, t_max, int(sample_rate * t_max))
    H_td = [Qobj(H), [Qobj(mu), drive_func]] 
    result = mesolve(H_td, psi0, tlist, c_ops=Ops, e_ops=Obs, args=args)
    signal = result.expect[0]
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    return tlist, signal

def optimize_control_pulse_3d(H_np, mu_x_np, mu_y_np, mu_z_np, psi0_np, target_psi_np, t_max, n_steps=100, epochs=50, lr=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = t_max / n_steps
    H = torch.tensor(H_np, dtype=torch.complex64, device=device)
    mu_x = torch.tensor(mu_x_np, dtype=torch.complex64, device=device)
    mu_y = torch.tensor(mu_y_np, dtype=torch.complex64, device=device)
    mu_z = torch.tensor(mu_z_np, dtype=torch.complex64, device=device)
    psi0 = torch.tensor(psi0_np, dtype=torch.complex64, device=device).unsqueeze(1)
    target_psi = torch.tensor(target_psi_np, dtype=torch.complex64, device=device).unsqueeze(1)
    
    u_x = torch.zeros(n_steps, requires_grad=True, device=device)
    u_y = torch.zeros(n_steps, requires_grad=True, device=device)
    u_z = torch.zeros(n_steps, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([u_x, u_y, u_z], lr=lr)
    
    progress_bar = st.progress(0)
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        psi = psi0
        for i in range(n_steps):
            H_tot = H + u_x[i]*mu_x + u_y[i]*mu_y + u_z[i]*mu_z
            U = torch.matrix_exp(-1j * dt * H_tot)
            psi = torch.matmul(U, psi)
            
        overlap = torch.abs(torch.vdot(target_psi.squeeze(), psi.squeeze()))**2
        loss = 1.0 - overlap
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        
    return (u_x.detach().cpu().numpy(), u_y.detach().cpu().numpy(), u_z.detach().cpu().numpy()), loss_history, np.linspace(0, t_max, n_steps), psi.detach().cpu().numpy().flatten()

# --- 3. MODULAR DATA LOADER (No m.h5) ---
@st.cache_data
def load_system(n_val):
    """Loads Hamiltonian and Dipole matrices from the N-specific directory."""
    base_dir = f"{n_val}/"
    try:
        with h5py.File(f'{base_dir}main.h5', 'r') as h_file: 
            H = np.array(h_file['main_hamiltonian'])
        with h5py.File(f'{base_dir}x.h5', 'r') as x_file:
            mu_x = np.array(x_file['dipole-x']) * 1j
        with h5py.File(f'{base_dir}y.h5', 'r') as y_file:
            mu_y = np.array(y_file['dipole-y']) * 1j
        with h5py.File(f'{base_dir}z.h5', 'r') as z_file:
            mu_z = np.array(z_file['dipole-z']) * 1j
            
        dipoles = {'X-Axis': mu_x, 'Y-Axis': mu_y, 'Z-Axis': mu_z}
        return H, dipoles
    except Exception as e:
        print(f"CRITICAL ERROR loading N={n_val}: {repr(e)}")
        return None, None

# --- 4. SIDEBAR SETTINGS ---
st.sidebar.header("System Settings")

available_ns = [d for d in os.listdir('.') if os.path.isdir(d) and d.isdigit()]
available_ns.sort(key=int)
if not available_ns:
    available_ns = ["1"] # Fallback

active_n = st.sidebar.selectbox("Active System (N):", options=available_ns)

# Load the raw, full-sized matrices
H_raw, dipoles_raw = load_system(active_n)

if H_raw is not None and dipoles_raw is not None:
    original_size = H_raw.shape[0]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Matrix Truncation")
    st.sidebar.write(f"Raw Matrix Size: {original_size}x{original_size}")
    
    # --- THE TRUNCATOR ---
    # Default to 20 states, or the max size if it's smaller than 20
    trunc_size = st.sidebar.slider("Energy Levels to Include:", min_value=2, max_value=original_size, value=min(20, original_size), step=1)
    
    # Slice the Hamiltonian and Dipoles down to the selected size
    H_matrix = H_raw[:trunc_size, :trunc_size]
    dipole_dict = {
        'X-Axis': dipoles_raw['X-Axis'][:trunc_size, :trunc_size],
        'Y-Axis': dipoles_raw['Y-Axis'][:trunc_size, :trunc_size],
        'Z-Axis': dipoles_raw['Z-Axis'][:trunc_size, :trunc_size]
    }
    
    st.sidebar.success(f"✅ Matrices Truncated to {trunc_size}x{trunc_size}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Laser Polarization")
    selected_axis = st.sidebar.radio("Select Drive/Measurement Axis:", ['X-Axis', 'Y-Axis', 'Z-Axis'])
    mu_matrix = dipole_dict[selected_axis]
else:
    st.sidebar.error(f"Error reading files from directory '{active_n}/'. Check terminal for details.")
    H_matrix, dipole_dict, mu_matrix = None, None, None
# --- 5. CREATE THE TABS ---
tab1, tab2, tab3 = st.tabs(["🎛️ The Synthesizer", "🎚️ The Paddle Board", "🧠 3D PyTorch Control"])#, "🔍 N vs N' Differencing"])

if H_matrix is not None and dipole_dict is not None:
    energies, states = np.linalg.eigh(H_matrix)
    n_states = len(energies)

    # ==========================================
    # TAB 1: THE SYNTHESIZER
    # ==========================================
    with tab1:
        st.markdown(f"### Dynamics Controls (N={active_n}, listening to {selected_axis})")
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            initial_state_idx = st.number_input("Initial State |ψ₀⟩", min_value=0, max_value=n_states-1, value=0, key="t1_init")
        with col_2:
            drive_freq = st.number_input("Drive Frequency (ω)", value=1.0, step=0.1)
        with col_3:
            drive_amp = st.number_input("Drive Amplitude (A)", value=0.5, step=0.1)

        st.markdown("### Audio Controls")
        col_a, col_b = st.columns(2)
        with col_a:
            t_max_input = st.number_input("Signal Duration (Atomic Time)", min_value=0.1, max_value=600.0, value=60.0, step=0.5)
        with col_b:
            sample_rate_input = st.selectbox("Sample Rate (Resolution)", options=[100, 220, 441, 960], index=0)

        if st.button("Simulate Dynamics", type="primary"):
            with st.spinner("Solving Master Equation..."):
                psi0_vec = states[:, initial_state_idx]
                
                def drive(t, args): return args['A'] * np.cos(args['w'] * t)
                
                t, audio_out = compute_quantum_signal(
                    H_matrix, mu_matrix, Ops=[], Obs=[Qobj(mu_matrix)], psi0=Qobj(psi0_vec), 
                    drive_func=drive, args={'A': drive_amp, 'w': drive_freq}, t_max=t_max_input, sample_rate=sample_rate_input
                )
                
                fig = go.Figure(data=[go.Scatter(x=t, y=audio_out, mode='lines', name='⟨μ⟩ Expectation', line=dict(color='teal', width=1))])
                fig.update_layout(title="Time-Domain Radiation", xaxis_title="Time (Atomic Units)", yaxis_title="⟨μ⟩", template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                audio_data = np.int16(audio_out * 32767)
                virtual_file = io.BytesIO()
                wavfile.write(virtual_file, sample_rate_input, audio_data)
                st.audio(virtual_file, format="audio/wav")

    # ==========================================
    # TAB 2: THE PADDLE BOARD
    # ==========================================
    with tab2:
        st.markdown(f"### 🎚️ 3D Paddle Board (N={active_n})")
        
        max_x = np.max(np.abs(states.T.conj() @ dipole_dict['X-Axis'] @ states))
        max_y = np.max(np.abs(states.T.conj() @ dipole_dict['Y-Axis'] @ states))
        max_z = np.max(np.abs(states.T.conj() @ dipole_dict['Z-Axis'] @ states))
        global_max = max(max_x, max_y, max_z)
        
        ket_idx = st.number_input("Select Ket |j⟩ for Scattering", min_value=0, max_value=n_states-1, value=0)
        ket_vec = states[:, ket_idx]
            
        fig_bar = go.Figure(data=[
            go.Bar(name='X-Axis', x=[f"|{i}⟩" for i in range(n_states)], y=np.abs(states.T.conj() @ (dipole_dict['X-Axis'] @ ket_vec)).flatten(), marker_color='red'),
            go.Bar(name='Y-Axis', x=[f"|{i}⟩" for i in range(n_states)], y=np.abs(states.T.conj() @ (dipole_dict['Y-Axis'] @ ket_vec)).flatten(), marker_color='green'),
            go.Bar(name='Z-Axis', x=[f"|{i}⟩" for i in range(n_states)], y=np.abs(states.T.conj() @ (dipole_dict['Z-Axis'] @ ket_vec)).flatten(), marker_color='blue')
        ])
        fig_bar.update_layout(barmode='group', yaxis=dict(range=[0, global_max * 1.1]), xaxis_title="Eigenstates", yaxis_title="Amplitude Projection", template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # TAB 3: PYTORCH 3D OPTIMAL CONTROL
    # ==========================================
    with tab3:
        st.markdown(f"### 🧠 3D Quantum Pulse Optimization (N={active_n})")
        
        col_s, col_t, col_e, col_d = st.columns(4)
        with col_s: init_idx = st.number_input("Start |ψ₀⟩", min_value=0, max_value=n_states-1, value=0)
        with col_t: targ_idx = st.number_input("Target |ψ_target⟩", min_value=0, max_value=n_states-1, value=1)
        with col_e: q_epochs = st.number_input("Epochs", min_value=10, max_value=500, value=50, step=10)
        with col_d: q_dur = st.number_input("Duration (a.u.)", min_value=1.0, max_value=200.0, value=10.0, step=1.0)
            
        if st.button("Sculpt 3D Pulse", type="primary"):
            with st.spinner("Sculpting..."):
                psi0_vec_pt = states[:, init_idx]
                target_vec_pt = states[:, targ_idx]
                
                u_pulses, loss_h, t_p, final_psi = optimize_control_pulse_3d(
                    H_matrix, dipole_dict['X-Axis'], dipole_dict['Y-Axis'], dipole_dict['Z-Axis'], 
                    psi0_vec_pt, target_vec_pt, t_max=q_dur, n_steps=200, epochs=q_epochs
                )
                u_x, u_y, u_z = u_pulses
                
                initial_pops = np.abs(states.T.conj() @ psi0_vec_pt)**2
                final_pops = np.abs(states.T.conj() @ final_psi)**2
                
                fig_pops = go.Figure(data=[
                    go.Bar(name='Initial State', x=[f"|{i}⟩" for i in range(n_states)], y=initial_pops, marker_color='teal'),
                    go.Bar(name='Final State', x=[f"|{i}⟩" for i in range(n_states)], y=final_pops, marker_color='magenta')
                ])
                fig_pops.update_layout(title="State Occupation Dynamics", xaxis_title="Eigenstates", yaxis_title="|⟨j|ψ⟩|²", barmode='group', template="plotly_dark", height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_pops, use_container_width=True)

                fig_loss = go.Figure(data=[go.Scatter(y=loss_h, mode='lines', line=dict(color='magenta'))])
                fig_loss.update_layout(title="Training Infidelity", xaxis_title="Epoch", yaxis_title="1 - Fidelity", template="plotly_dark", height=250, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_loss, use_container_width=True)
                
                fig_pulse = go.Figure()
                fig_pulse.add_trace(go.Scatter(x=t_p, y=u_x, mode='lines', name='X-Drive', line=dict(color='red')))
                fig_pulse.add_trace(go.Scatter(x=t_p, y=u_y, mode='lines', name='Y-Drive', line=dict(color='green')))
                fig_pulse.add_trace(go.Scatter(x=t_p, y=u_z, mode='lines', name='Z-Drive', line=dict(color='blue')))
                fig_pulse.update_layout(title="3D Optimized Field", xaxis_title="Time (a.u.)", yaxis_title="Amplitude", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_pulse, use_container_width=True)
                
                st.success("Pulse Optimized! Audio generation simulated below.")

#     # ==========================================
#     # TAB 4: 🔍 N vs N' DIFFERENCING
#     # ==========================================
#     with tab4:
#         st.markdown("### 🔍 Correlation Differencing")
#         st.info("Momentum spatial visualization is currently disconnected while updating the HDF5 matrices. Check back soon!")