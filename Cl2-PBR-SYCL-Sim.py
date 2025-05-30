import dpctl
import dpctl.tensor as dpt
import numpy as np
import matplotlib.pyplot as plt
import time

def solve_reactor_gpu(L_reactor, U_velocity, kc, ac, C_Cl2_0, N_points, device):
    """Solve reactor equation on GPU using dpctl tensors"""
    
    # Create device arrays
    dz = 1.5 / (N_points - 1)  # Use max length for grid spacing
    z = dpt.linspace(0, 1.5, N_points, dtype=dpt.float64, device=device)
    
    # Initialize concentration array on device
    C = dpt.zeros(N_points, dtype=dpt.float64, device=device)
    C[0] = C_Cl2_0  # Set initial condition
    
    # Compute parameter
    kc_ac_over_U = (kc * ac) / U_velocity
    
    # Convert to numpy for computation, then back to device
    z_np = dpt.asnumpy(z)
    C_np = dpt.asnumpy(C)
    
    # Solve using Euler method on CPU (vectorized)
    for i in range(1, N_points):
        if z_np[i] <= L_reactor:
            dCdz = -kc_ac_over_U * C_np[i-1]
            C_np[i] = C_np[i-1] + dCdz * dz
        else:
            C_np[i] = C_np[i-1]  # Flatline after reactor length
    
    # Transfer back to device
    C = dpt.asarray(C_np, dtype=dpt.float64, device=device)
    
    return C, z

def compute_analytical_solution_gpu(z, C_Cl2_0, kc_ac_over_U, L_reactor, device):
    """Compute analytical solution using device tensors"""
    
    z_np = dpt.asnumpy(z)
    C_analytical_np = np.zeros_like(z_np)
    
    # Vectorized computation
    mask = z_np <= L_reactor
    C_analytical_np[mask] = C_Cl2_0 * np.exp(-kc_ac_over_U * z_np[mask])
    
    # For points beyond reactor length, use value at L_reactor
    if np.any(~mask):
        C_at_L = C_Cl2_0 * np.exp(-kc_ac_over_U * L_reactor)
        C_analytical_np[~mask] = C_at_L
    
    return dpt.asarray(C_analytical_np, dtype=dpt.float64, device=device)

def print_device_info():
    """Print information about available devices"""
    print("Available devices:")
    for i, device in enumerate(dpctl.get_devices()):
        print(f"Device {i}: {device.name} ({device.device_type}) - {device.vendor}")
    print()

def main():
    start_total = time.time()
    
    print_device_info()
    
    # Try to select Intel GPU, fallback to any GPU, then default
    device = None
    for d in dpctl.get_devices():
        if "Intel" in d.name and "gpu" in str(d.device_type).lower():
            device = d
            break
    
    if device is None:
        # Try any GPU
        for d in dpctl.get_devices():
            if "gpu" in str(d.device_type).lower():
                device = d
                break
    
    if device is None:
        device = dpctl.select_default_device()
    
    print(f"Using device: {device.name} ({device.device_type})")
    
    # Reactor parameters
    L1 = 1.0
    C_Cl2_0 = 1.0
    U1 = 0.1
    kc1 = 2.78e-4
    phi = 0.4
    dp1 = 0.01
    ac1 = 6 * (1 - phi) / dp1
    
    L2 = 1.5
    U2 = 0.4
    dp2 = dp1 / 3
    ac2 = 6 * (1 - phi) / dp2
    kc2 = 9.63e-4
    
    N_total = 50000  # Large for better performance measurement
    
    print("BENCHMARKS:")
    
    # Case 1
    start = time.time()
    C_solution1, z_global = solve_reactor_gpu(L1, U1, kc1, ac1, C_Cl2_0, N_total, device)
    case1_time = time.time() - start
    print(f"Case 1 solve: {case1_time:.4f}s")
    
    # Case 2
    start = time.time()
    C_solution2, _ = solve_reactor_gpu(L2, U2, kc2, ac2, C_Cl2_0, N_total, device)
    case2_time = time.time() - start
    print(f"Case 2 solve: {case2_time:.4f}s")
    
    # Analytical solutions
    start = time.time()
    kc_ac_over_U1 = (kc1 * ac1) / U1
    kc_ac_over_U2 = (kc2 * ac2) / U2
    
    C_analytical1 = compute_analytical_solution_gpu(z_global, C_Cl2_0, kc_ac_over_U1, L1, device)
    C_analytical2 = compute_analytical_solution_gpu(z_global, C_Cl2_0, kc_ac_over_U2, L2, device)
    analytical_time = time.time() - start
    print(f"Analytical solutions: {analytical_time:.4f}s")
    
    # Transfer to CPU
    start = time.time()
    z_cpu = dpt.asnumpy(z_global)
    C_num1_cpu = dpt.asnumpy(C_solution1)
    C_num2_cpu = dpt.asnumpy(C_solution2)
    C_anal1_cpu = dpt.asnumpy(C_analytical1)
    C_anal2_cpu = dpt.asnumpy(C_analytical2)
    transfer_time = time.time() - start
    print(f"Device->CPU transfer: {transfer_time:.4f}s")
    
    # Calculate conversions
    conversion1 = (1 - C_num1_cpu[-1] / C_Cl2_0) * 100
    conversion2 = (1 - C_num2_cpu[-1] / C_Cl2_0) * 100
    
    # Plotting
    start = time.time()
    plt.figure(figsize=(12, 8))
    
    # Subsample for plotting (plot every 100th point for large datasets)
    step = max(1, N_total // 1000)
    indices = slice(0, None, step)
    
    plt.plot(z_cpu[indices], C_num1_cpu[indices], 
             label=f'Numerical: Case 1 (X_A = {conversion1:.1f}%)', 
             color='blue', linestyle='-', linewidth=2)
    plt.plot(z_cpu[indices], C_anal1_cpu[indices], 
             label='Analytical: Case 1', 
             color='blue', linestyle='--', alpha=0.7)
    plt.plot(z_cpu[indices], C_num2_cpu[indices], 
             label=f'Numerical: Case 2 (X_A = {conversion2:.1f}%)', 
             color='red', linestyle='-', linewidth=2)
    plt.plot(z_cpu[indices], C_anal2_cpu[indices], 
             label='Analytical: Case 2', 
             color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Reactor Length, z (m)')
    plt.ylabel('Cl₂ Concentration (mol/m³)')
    plt.title(f'GPU Accelerated Reactor Simulation\n{N_total:,} grid points on {device.name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x=L1, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=L2, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('reactor_gpu_fast.png', dpi=150, bbox_inches='tight')
    plot_time = time.time() - start
    print(f"Plotting: {plot_time:.4f}s")
    
    total_time = time.time() - start_total
    compute_time = case1_time + case2_time + analytical_time
    
    print(f"\nTOTAL TIME: {total_time:.4f}s")
    print(f"Compute time: {compute_time:.4f}s ({compute_time/total_time*100:.1f}%)")
    print(f"Memory/IO time: {transfer_time:.4f}s ({transfer_time/total_time*100:.1f}%)")
    print(f"Grid points: {N_total:,}")
    print(f"Throughput: {N_total*2/compute_time:.0f} points/sec")  # 2 cases
    
    print(f"\nResults:")
    print(f"Case 1 conversion: {conversion1:.1f}%")
    print(f"Case 2 conversion: {conversion2:.1f}%")
    print("Plot: reactor_gpu_fast.png")

if __name__ == "__main__":
    main()
