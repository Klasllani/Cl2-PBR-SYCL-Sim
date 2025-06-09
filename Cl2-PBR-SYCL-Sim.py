import dpctl
import dpctl.tensor as dpt
import numpy as np
import matplotlib.pyplot as plt
import time

def solve_reactor_vectorized_gpu(L_reactor, U_velocity, kc, ac, C_Cl2_0, N_points, device):
    """Solve reactor equation using vectorized GPU computation"""
    
    # Create device arrays
    dz = 1.5 / (N_points - 1)
    z = dpt.linspace(0, 1.5, N_points, dtype=dpt.float64, device=device)
    
    # Compute parameter on device
    kc_ac_over_U = (kc * ac) / U_velocity
    
    # Vectorized solution: C(z) = C0 * exp(-k*z) for z <= L_reactor
    # Create mask for points within reactor
    mask_reactor = z <= L_reactor
    
    # Initialize concentration array
    C = dpt.zeros(N_points, dtype=dpt.float64, device=device)
    
    # Compute exponential decay within reactor using vectorized operations
    z_reactor = dpt.where(mask_reactor, z, 0.0)  # Zero out points beyond reactor
    exp_term = dpt.exp(-kc_ac_over_U * z_reactor)
    C = dpt.where(mask_reactor, C_Cl2_0 * exp_term, 0.0)
    
    # For points beyond reactor, use concentration at reactor exit
    if L_reactor < 1.5:  # Only if reactor doesn't span full length
        C_exit = C_Cl2_0 * dpt.exp(-kc_ac_over_U * L_reactor)
        C = dpt.where(~mask_reactor, C_exit, C)
    
    return C, z

def solve_reactor_vectorized_cpu(L_reactor, U_velocity, kc, ac, C_Cl2_0, N_points):
    """Solve reactor equation using vectorized CPU computation"""
    
    # Create arrays
    dz = 1.5 / (N_points - 1)
    z = np.linspace(0, 1.5, N_points, dtype=np.float64)
    
    # Compute parameter
    kc_ac_over_U = (kc * ac) / U_velocity
    
    # Vectorized solution
    mask_reactor = z <= L_reactor
    
    # Initialize concentration array
    C = np.zeros(N_points, dtype=np.float64)
    
    # Compute exponential decay within reactor
    C[mask_reactor] = C_Cl2_0 * np.exp(-kc_ac_over_U * z[mask_reactor])
    
    # For points beyond reactor, use concentration at reactor exit
    if L_reactor < 1.5:
        C_exit = C_Cl2_0 * np.exp(-kc_ac_over_U * L_reactor)
        C[~mask_reactor] = C_exit
    
    return C, z

def solve_reactor_euler_gpu(L_reactor, U_velocity, kc, ac, C_Cl2_0, N_points, device):
    """Solve reactor using Euler method on GPU (for comparison/validation)"""
    
    dz = 1.5 / (N_points - 1)
    z = dpt.linspace(0, 1.5, N_points, dtype=dpt.float64, device=device)
    kc_ac_over_U = (kc * ac) / U_velocity
    
    # For Euler method, we need to do this iteratively
    # Convert to numpy for the loop (this is inherently serial)
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

def compute_analytical_solution(z, C_Cl2_0, kc_ac_over_U, L_reactor, use_gpu=False, device=None):
    """Compute analytical solution"""
    
    if use_gpu and device is not None:
        # GPU version
        if isinstance(z, np.ndarray):
            z = dpt.asarray(z, dtype=dpt.float64, device=device)
        
        mask = z <= L_reactor
        C_analytical = dpt.zeros_like(z)
        
        # Within reactor
        C_analytical = dpt.where(mask, C_Cl2_0 * dpt.exp(-kc_ac_over_U * z), C_analytical)
        
        # Beyond reactor
        if L_reactor < 1.5:
            C_exit = C_Cl2_0 * dpt.exp(-kc_ac_over_U * L_reactor)
            C_analytical = dpt.where(~mask, C_exit, C_analytical)
        
        return C_analytical
    else:
        # CPU version
        if hasattr(z, 'asnumpy'):  # Convert from dpctl tensor if needed
            z = dpt.asnumpy(z)
        
        mask = z <= L_reactor
        C_analytical = np.zeros_like(z)
        
        # Within reactor
        C_analytical[mask] = C_Cl2_0 * np.exp(-kc_ac_over_U * z[mask])
        
        # Beyond reactor
        if L_reactor < 1.5:
            C_exit = C_Cl2_0 * np.exp(-kc_ac_over_U * L_reactor)
            C_analytical[~mask] = C_exit
        
        return C_analytical

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
        solve_func = solve_reactor_vectorized_gpu
        solve_euler_func = solve_reactor_euler_gpu
    else:
        print("Using CPU (no suitable GPU found)")
        solve_func = solve_reactor_vectorized_cpu
        solve_euler_func = solve_reactor_euler_cpu
    
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
    
    # Case 1 - Vectorized solution
    start = time.time()
    if use_gpu:
        C_solution1, z_global = solve_func(L1, U1, kc1, ac1, C_Cl2_0, N_total, device)
    else:
        C_solution1, z_global = solve_func(L1, U1, kc1, ac1, C_Cl2_0, N_total)
    case1_time = time.time() - start
    print(f"Case 1 (vectorized): {case1_time:.4f}s")
    
    # Case 2 - Vectorized solution
    start = time.time()
    if use_gpu:
        C_solution2, _ = solve_func(L2, U2, kc2, ac2, C_Cl2_0, N_total, device)
    else:
        C_solution2, _ = solve_func(L2, U2, kc2, ac2, C_Cl2_0, N_total)
    case2_time = time.time() - start
    print(f"Case 2 (vectorized): {case2_time:.4f}s")
    
    # Analytical solutions
    start = time.time()
    kc_ac_over_U1 = (kc1 * ac1) / U1
    kc_ac_over_U2 = (kc2 * ac2) / U2
    
    C_analytical1 = compute_analytical_solution(z_global, C_Cl2_0, kc_ac_over_U1, L1, use_gpu, device)
    C_analytical2 = compute_analytical_solution(z_global, C_Cl2_0, kc_ac_over_U2, L2, use_gpu, device)
    analytical_time = time.time() - start
    print(f"Analytical solutions: {analytical_time:.4f}s")
    
    # Euler method comparison (smaller grid for speed)
    N_euler = 5000
    start = time.time()
    if use_gpu:
        C_euler1, z_euler = solve_euler_func(L1, U1, kc1, ac1, C_Cl2_0, N_euler, device)
        C_euler2, _ = solve_euler_func(L2, U2, kc2, ac2, C_Cl2_0, N_euler, device)
    else:
        C_euler1, z_euler = solve_euler_func(L1, U1, kc1, ac1, C_Cl2_0, N_euler)
        C_euler2, _ = solve_euler_func(L2, U2, kc2, ac2, C_Cl2_0, N_euler)
    euler_time = time.time() - start
    print(f"Euler method ({N_euler:,} points): {euler_time:.4f}s")
    
    # Convert to numpy for plotting
    start = time.time()
    if use_gpu:
        z_cpu = dpt.asnumpy(z_global)
        C_num1_cpu = dpt.asnumpy(C_solution1)
        C_num2_cpu = dpt.asnumpy(C_solution2)
        C_anal1_cpu = dpt.asnumpy(C_analytical1)
        C_anal2_cpu = dpt.asnumpy(C_analytical2)
        z_euler_cpu = dpt.asnumpy(z_euler)
        C_euler1_cpu = dpt.asnumpy(C_euler1)
        C_euler2_cpu = dpt.asnumpy(C_euler2)
    else:
        z_cpu = z_global
        C_num1_cpu = C_solution1
        C_num2_cpu = C_solution2
        C_anal1_cpu = C_analytical1
        C_anal2_cpu = C_analytical2
        z_euler_cpu = z_euler
        C_euler1_cpu = C_euler1
        C_euler2_cpu = C_euler2
    
    transfer_time = time.time() - start
    if use_gpu:
        print(f"GPU->CPU transfer: {transfer_time:.4f}s")
    
    # Calculate conversions using Euler method results
    conversion1 = (1 - C_euler1_cpu[-1] / C_Cl2_0) * 100
    conversion2 = (1 - C_euler2_cpu[-1] / C_Cl2_0) * 100
    
    # Plotting
    start = time.time()
    plt.figure(figsize=(12, 8))
    
    # Subsample for plotting
    step = max(1, N_euler // 1000)
    indices = slice(0, None, step)
    
    # Use Euler method results (original approach)
    plt.plot(z_euler_cpu[indices], C_euler1_cpu[indices], 
             label=f'Numerical: Case 1 (X_A = {conversion1:.1f}%)', 
             color='blue', linestyle='-', linewidth=2)
    plt.plot(z_euler_cpu[indices], C_anal1_cpu[::N_total//N_euler][indices], 
             label='Analytical: Case 1', 
             color='blue', linestyle='--', alpha=0.7)
    plt.plot(z_euler_cpu[indices], C_euler2_cpu[indices], 
             label=f'Numerical: Case 2 (X_A = {conversion2:.1f}%)', 
             color='red', linestyle='-', linewidth=2)
    plt.plot(z_euler_cpu[indices], C_anal2_cpu[::N_total//N_euler][indices], 
             label='Analytical: Case 2', 
             color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Reactor Length, z (m)')
    plt.ylabel('Cl₂ Concentration (mol/m³)')
    plt.title(f'{"GPU" if use_gpu else "CPU"} Accelerated Reactor Simulation\n{N_euler:,} grid points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x=L1, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=L2, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('output.png', dpi=150, bbox_inches='tight')
    plot_time = time.time() - start
    print(f"Plotting: {plot_time:.4f}s")
    
    total_time = time.time() - start_total
    compute_time = case1_time + case2_time + analytical_time
    
    print(f"\nPERFORMACE SUMMARY:")
    print(f"Total time: {total_time:.4f}s")
    print(f"Vectorized compute: {compute_time:.4f}s ({compute_time/total_time*100:.1f}%)")
    print(f"Euler method: {euler_time:.4f}s")
    if use_gpu:
        print(f"Memory transfer: {transfer_time:.4f}s ({transfer_time/total_time*100:.1f}%)")
    print(f"Grid points (vectorized): {N_total:,}")
    print(f"Grid points (Euler): {N_euler:,}")
    print(f"Vectorized throughput: {N_total*2/compute_time:.0f} points/sec")
    
    print(f"\nRESULTS:")
    print(f"Case 1 conversion: {conversion1:.1f}%")
    print(f"Case 2 conversion: {conversion2:.1f}%")
    print("Plot saved: output.png")
    
    # Verify accuracy (compare Euler with analytical at same grid resolution)
    C_anal1_euler = compute_analytical_solution(z_euler, C_Cl2_0, kc_ac_over_U1, L1, use_gpu, device)
    C_anal2_euler = compute_analytical_solution(z_euler, C_Cl2_0, kc_ac_over_U2, L2, use_gpu, device)
    
    if use_gpu:
        C_anal1_euler_cpu = dpt.asnumpy(C_anal1_euler)
        C_anal2_euler_cpu = dpt.asnumpy(C_anal2_euler)
    else:
        C_anal1_euler_cpu = C_anal1_euler
        C_anal2_euler_cpu = C_anal2_euler
    
    max_error1 = np.max(np.abs(C_euler1_cpu - C_anal1_euler_cpu))
    max_error2 = np.max(np.abs(C_euler2_cpu - C_anal2_euler_cpu))
    print(f"\nACCURACY:")
    print(f"Max error Case 1: {max_error1:.2e}")
    print(f"Max error Case 2: {max_error2:.2e}")

if __name__ == "__main__":
    main()
