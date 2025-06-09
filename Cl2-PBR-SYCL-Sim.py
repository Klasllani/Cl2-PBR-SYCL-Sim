import dpctl
import dpctl.tensor as dpt
import numpy as np
import matplotlib.pyplot as plt
import time

def solve_reactor_euler_gpu(L_reactor, U_velocity, kc, ac, C_Cl2_0, N_points, device):
    """Solve reactor using Euler method on GPU"""
    
    dz = 1.5 / (N_points - 1)
    z = dpt.linspace(0, 1.5, N_points, dtype=dpt.float64, device=device)
    kc_ac_over_U = (kc * ac) / U_velocity
    
    # Iteration
    z_np = dpt.asnumpy(z)
    C_np = np.zeros(N_points, dtype=np.float64)
    C_np[0] = C_Cl2_0
    
    # Euler integration
    for i in range(1, N_points):
        if z_np[i] <= L_reactor:
            dCdz = -kc_ac_over_U * C_np[i-1]
            C_np[i] = C_np[i-1] + dCdz * dz
        else:
            C_np[i] = C_np[i-1]  # Constant beyond reactor
    
    # Transfer back to device
    C = dpt.asarray(C_np, dtype=np.float64, device=device)
    
    return C, z

def solve_reactor_euler_cpu(L_reactor, U_velocity, kc, ac, C_Cl2_0, N_points):
    """Solve reactor using Euler method on CPU"""
    
    dz = 1.5 / (N_points - 1)
    z = np.linspace(0, 1.5, N_points, dtype=np.float64)
    kc_ac_over_U = (kc * ac) / U_velocity
    
    C = np.zeros(N_points, dtype=np.float64)
    C[0] = C_Cl2_0
    
    # Euler integration
    for i in range(1, N_points):
        if z[i] <= L_reactor:
            dCdz = -kc_ac_over_U * C[i-1]
            C[i] = C[i-1] + dCdz * dz
        else:
            C[i] = C[i-1]  # Constant beyond reactor
    
    return C, z

def get_best_device():
    """Get the best available device, defaulting to CPU"""
    try:
        # Try to get available devices
        devices = dpctl.get_devices()
        
        # Prefer Intel GPU
        for device in devices:
            if "Intel" in device.name and "gpu" in str(device.device_type).lower():
                return device, True
        
        # Then any GPU
        for device in devices:
            if "gpu" in str(device.device_type).lower():
                return device, True
        
        # Finally, try default device
        try:
            default_device = dpctl.select_default_device()
            if "gpu" in str(default_device.device_type).lower():
                return default_device, True
        except:
            pass
            
    except Exception as e:
        print(f"GPU detection failed: {e}")
    
    # Fall back to CPU
    return None, False

def print_device_info():
    """Print information about available devices"""
    try:
        print("Available devices:")
        for i, device in enumerate(dpctl.get_devices()):
            print(f"  Device {i}: {device.name} ({device.device_type}) - {device.vendor}")
    except Exception as e:
        print(f"Could not enumerate devices: {e}")
    print()

def main():
    start_total = time.time()
    
    print_device_info()
    
    # Get best available device
    device, use_gpu = get_best_device()
    
    if use_gpu:
        print(f"Using GPU: {device.name} ({device.device_type})")
        solve_func = solve_reactor_euler_gpu
        device_label = f"GPU: {device.name}"
    else:
        print("Using CPU (no suitable GPU found)")
        solve_func = solve_reactor_euler_cpu
        device_label = "CPU"
    
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
    
    N_total = 50000
    
    print("BENCHMARKS:")
    
    # Case 1 solving
    start = time.time()
    if use_gpu:
        C_solution1, z_global = solve_func(L1, U1, kc1, ac1, C_Cl2_0, N_total, device)
    else:
        C_solution1, z_global = solve_func(L1, U1, kc1, ac1, C_Cl2_0, N_total)
    case1_time = time.time() - start
    print(f"Case 1: {case1_time:.4f}s")
    
    # Case 2 solving
    start = time.time()
    if use_gpu:
        C_solution2, _ = solve_func(L2, U2, kc2, ac2, C_Cl2_0, N_total, device)
    else:
        C_solution2, _ = solve_func(L2, U2, kc2, ac2, C_Cl2_0, N_total)
    case2_time = time.time() - start
    print(f"Case 2: {case2_time:.4f}s")
    
    # Convert to numpy for plotting
    start = time.time()
    if use_gpu:
        z_cpu = dpt.asnumpy(z_global)
        C_num1_cpu = dpt.asnumpy(C_solution1)
        C_num2_cpu = dpt.asnumpy(C_solution2)
    else:
        z_cpu = z_global
        C_num1_cpu = C_solution1
        C_num2_cpu = C_solution2
    
    transfer_time = time.time() - start
    if use_gpu:
        print(f"GPU->CPU transfer: {transfer_time:.4f}s")
    
    # Calculate conversions
    conversion1 = (1 - C_num1_cpu[-1] / C_Cl2_0) * 100
    conversion2 = (1 - C_num2_cpu[-1] / C_Cl2_0) * 100
    
    # Plotting
    start = time.time()
    plt.figure(figsize=(12, 8))
    
    # Subsample for plotting
    step = max(1, N_total // 1000)
    indices = slice(0, None, step)
    
    plt.plot(z_cpu[indices], C_num1_cpu[indices], 
             label=f'Case 1: L = {L1} m (X_A = {conversion1:.1f}%)', 
             color='blue', linestyle='-', linewidth=2)
    plt.plot(z_cpu[indices], C_num2_cpu[indices], 
             label=f'Case 2: L = {L2} m (X_A = {conversion2:.1f}%)', 
             color='red', linestyle='-', linewidth=2)
    
    plt.xlabel('Reactor Length, z (m)')
    plt.ylabel('Cl₂ Concentration (mol/m³)')
    plt.title('Chlorine Concentration Profile Along Packed Bed Reactor Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines to show reactor lengths
    plt.axvline(x=L1, color='blue', linestyle=':', alpha=0.5, label='_nolegend_')
    plt.axvline(x=L2, color='red', linestyle=':', alpha=0.5, label='_nolegend_')
    
    # Add device info in corner
    plt.text(0.98, 0.98, f'Computed on {device_label}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('output.png', dpi=150, bbox_inches='tight')
    plot_time = time.time() - start
    print(f"Plotting: {plot_time:.4f}s")
    
    total_time = time.time() - start_total
    compute_time = case1_time + case2_time
    
    print(f"\nPERFORMACE SUMMARY:")
    print(f"Total time: {total_time:.4f}s")
    print(f"Compute time: {compute_time:.4f}s ({compute_time/total_time*100:.1f}%)")
    if use_gpu:
        print(f"Memory transfer: {transfer_time:.4f}s ({transfer_time/total_time*100:.1f}%)")
    print(f"Grid points: {N_total:,}")
    print(f"Throughput: {N_total*2/compute_time:.0f} points/sec")
    
    print(f"\nRESULTS:")
    print(f"Case 1 conversion: {conversion1:.1f}%")
    print(f"Case 2 conversion: {conversion2:.1f}%")
    print("Plot saved: output.png")

if __name__ == "__main__":
    main()
