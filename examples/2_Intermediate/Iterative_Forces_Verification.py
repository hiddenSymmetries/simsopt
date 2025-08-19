import numpy as np
import simsoptpp

def test_abbv_force_calc_verification():
    """
    Test case to verify Iterative_Forces function by comparing with full dipole_forces_from_A_F calculation.
    
    Test scenario:
    1. Start with 10 magnets, all moments zero, random positions
    2. Build full A_F tensor from positions
    3. Progressively activate magnets and test Iterative_Forces vs dipole_forces_from_A_F
    """
    
    print("=== Abbv_Force_Calcs Verification Test ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Create 10 magnet array with all moments set to zero and randomized positions
    N = 10
    moments = np.zeros(3 * N)  # All moments zero
    positions = np.random.rand(3 * N) * 1  # Random positions between 0 and 1
    
    print(f"Testing with {N} magnets")
    
    # 2. Build full scale A_F tensor from positions
    A_F = simsoptpp.build_A_F_tensor(positions)
    
    # 3. Test j_index=11 (fourth magnet, z-component) - magnet NOT yet active
    j_index_1 = 11
    
    try:
        # Create initial forces array (all zeros)
        initial_forces = np.zeros(3 * N)
        
        # Call Iterative_Forces for first test - magnet 3 is NOT active yet
        forces_result_1, norm_result_1 = simsoptpp.Iterative_Forces(moments, initial_forces, j_index_1, positions)
        
        # NOW set the moment for comparison with dipole_forces_from_A_F
        moments[11] = 1.0
        
        # Calculate full forces for comparison
        full_forces_1 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_1 = simsoptpp.two_norm_squared(full_forces_1)
        
        print(f"adding magnet at index 11: Abbv={norm_result_1:.2e}, Full={full_norm_1:.2e}, Diff={abs(norm_result_1 - full_norm_1):.2e}")
        
        # 4. Test j_index=2 (first magnet, z-component)
        j_index_2 = 2
        forces_result_2, norm_result_2 = simsoptpp.Iterative_Forces(moments, forces_result_1, j_index_2, positions)
        moments[2] = 1.0
        full_forces_2 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_2 = simsoptpp.two_norm_squared(full_forces_2)
        print(f"adding magnet at index 2: Abbv={norm_result_2:.2e}, Full={full_norm_2:.2e}, Diff={abs(norm_result_2 - full_norm_2):.2e}")
        
        # 5. Test j_index=6 (third magnet, x-component)
        j_index_3 = 6
        forces_result_3, norm_result_3 = simsoptpp.Iterative_Forces(moments, forces_result_2, j_index_3, positions)
        moments[6] = 1.0
        full_forces_3 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_3 = simsoptpp.two_norm_squared(full_forces_3)
        print(f"adding magnet at index 6: Abbv={norm_result_3:.2e}, Full={full_norm_3:.2e}, Diff={abs(norm_result_3 - full_norm_3):.2e}")
        
        # 6. Test j_index=3 (second magnet, x-component) - OPPOSITE ORIENTATION
        j_index_4 = 3
        forces_result_4, norm_result_4 = simsoptpp.Iterative_Forces(moments, forces_result_3, j_index_4, positions, sign=-1)
        moments[3] = -1.0  # Negative orientation
        full_forces_4 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_4 = simsoptpp.two_norm_squared(full_forces_4)
        print(f"adding magnet at index 3: Abbv={norm_result_4:.2e}, Full={full_norm_4:.2e}, Diff={abs(norm_result_4 - full_norm_4):.2e}")
        
        # 7. Test j_index=25 (eighth magnet, y-component)
        j_index_5 = 25
        forces_result_5, norm_result_5 = simsoptpp.Iterative_Forces(moments, forces_result_4, j_index_5, positions)
        moments[25] = 1.0
        full_forces_5 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_5 = simsoptpp.two_norm_squared(full_forces_5)
        print(f"adding magnet at index 25: Abbv={norm_result_5:.2e}, Full={full_norm_5:.2e}, Diff={abs(norm_result_5 - full_norm_5):.2e}")
        
        # 8. Test j_index=17 (sixth magnet, z-component)
        j_index_6 = 17
        forces_result_6, norm_result_6 = simsoptpp.Iterative_Forces(moments, forces_result_5, j_index_6, positions)
        moments[17] = 1.0
        full_forces_6 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_6 = simsoptpp.two_norm_squared(full_forces_6)
        print(f"adding magnet at index 17: Abbv={norm_result_6:.2e}, Full={full_norm_6:.2e}, Diff={abs(norm_result_6 - full_norm_6):.2e}")
        
        # 9. Test j_index=13 (fifth magnet, y-component) - OPPOSITE ORIENTATION
        j_index_7 = 13
        forces_result_7, norm_result_7 = simsoptpp.Iterative_Forces(moments, forces_result_6, j_index_7, positions, sign=-1)
        moments[13] = -1.0  # Negative orientation
        full_forces_7 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_7 = simsoptpp.two_norm_squared(full_forces_7)
        print(f"adding magnet at index 13: Abbv={norm_result_7:.2e}, Full={full_norm_7:.2e}, Diff={abs(norm_result_7 - full_norm_7):.2e}")
        
        # 10. Test j_index=27 (tenth magnet, x-component) - OPPOSITE ORIENTATION
        j_index_8 = 27
        forces_result_8, norm_result_8 = simsoptpp.Iterative_Forces(moments, forces_result_7, j_index_8, positions, sign=-1)
        moments[27] = -1.0  # Negative orientation
        full_forces_8 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_8 = simsoptpp.two_norm_squared(full_forces_8)
        print(f"adding magnet at index 27: Abbv={norm_result_8:.2e}, Full={full_norm_8:.2e}, Diff={abs(norm_result_8 - full_norm_8):.2e}")
        
        # 11. Test j_index=21 (eighth magnet, x-component)
        j_index_9 = 21
        forces_result_9, norm_result_9 = simsoptpp.Iterative_Forces(moments, forces_result_8, j_index_9, positions)
        moments[21] = 1.0
        full_forces_9 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_9 = simsoptpp.two_norm_squared(full_forces_9)
        print(f"adding magnet at index 21: Abbv={norm_result_9:.2e}, Full={full_norm_9:.2e}, Diff={abs(norm_result_9 - full_norm_9):.2e}")
        
        # 12. Test j_index=20 (seventh magnet, z-component)
        j_index_10 = 20
        forces_result_10, norm_result_10 = simsoptpp.Iterative_Forces(moments, forces_result_9, j_index_10, positions)
        moments[20] = 1.0
        full_forces_10 = simsoptpp.dipole_forces_from_A_F(moments, A_F)
        full_norm_10 = simsoptpp.two_norm_squared(full_forces_10)
        print(f"adding magnet at index 20: Abbv={norm_result_10:.2e}, Full={full_norm_10:.2e}, Diff={abs(norm_result_10 - full_norm_10):.2e}")
        
        # 13. Summary of all tests
        print("\n=== FINAL SUMMARY ===")
        print(f"adding magnet at index 11 (magnet 3, z): {'PASS' if abs(norm_result_1 - full_norm_1) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 2  (magnet 0, z): {'PASS' if abs(norm_result_2 - full_norm_2) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 6  (magnet 2, x): {'PASS' if abs(norm_result_3 - full_norm_3) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 3  (magnet 1, -x): {'PASS' if abs(norm_result_4 - full_norm_4) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 25 (magnet 8, y): {'PASS' if abs(norm_result_5 - full_norm_5) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 17 (magnet 5, z): {'PASS' if abs(norm_result_6 - full_norm_6) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 13 (magnet 4, -y): {'PASS' if abs(norm_result_7 - full_norm_7) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 27 (magnet 9, -x): {'PASS' if abs(norm_result_8 - full_norm_8) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 21 (magnet 7, x): {'PASS' if abs(norm_result_9 - full_norm_9) < 1e-10 else 'FAIL'}")
        print(f"adding magnet at index 20 (magnet 6, z): {'PASS' if abs(norm_result_10 - full_norm_10) < 1e-10 else 'FAIL'}")
        
        all_tests_passed = (abs(norm_result_1 - full_norm_1) < 1e-10 and 
                           abs(norm_result_2 - full_norm_2) < 1e-10 and 
                           abs(norm_result_3 - full_norm_3) < 1e-10 and
                           abs(norm_result_4 - full_norm_4) < 1e-10 and
                           abs(norm_result_5 - full_norm_5) < 1e-10 and
                           abs(norm_result_6 - full_norm_6) < 1e-10 and
                           abs(norm_result_7 - full_norm_7) < 1e-10 and
                           abs(norm_result_8 - full_norm_8) < 1e-10 and
                           abs(norm_result_9 - full_norm_9) < 1e-10 and
                           abs(norm_result_10 - full_norm_10) < 1e-10)
        
        if all_tests_passed:
            print("\nALL TESTS PASSED! Iterative_Forces is working correctly.")
        else:
            print("\nSOME TESTS FAILED! There may be issues with Iterative_Forces.")
            
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_abbv_force_calc_verification()
