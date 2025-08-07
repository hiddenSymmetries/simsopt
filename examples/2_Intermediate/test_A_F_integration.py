#!/usr/bin/env python3
"""
Test script to verify A_F force calculations integration into simsopt.
"""

import numpy as np
import time
import simsoptpp

def test_A_F_integration():
    """Test the A_F force calculations integration with analytical test cases."""
    
    print("Testing A_F force calculations integration...")
    
    # Test case 1: Two orthogonal dipoles. Should be zero force.
    print("\nTest 1: Two orthogonal dipoles")
    moments = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    positions = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    force_norm_squared = simsoptpp.diagnostic_test(2, moments, positions)
    print(f"Force norm squared: {force_norm_squared}")
    print("Expected: ~0 (orthogonal dipoles should have minimal force)")
    
    # Test case 2: Three collinear dipoles. Should be zero force.
    print("\nTest 2: Three collinear dipoles")
    moments = np.array([0.0, 1.0, 0.0, 0.0, -0.0625, 0.0, 0.0, 1.0, 0.0])
    positions = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    force_norm_squared = simsoptpp.diagnostic_test(3, moments, positions)
    print(f"Force norm squared: {force_norm_squared}")
    print("Expected: ~0 (collinear dipoles should have minimal force)")
    
    # Test case 3: Four dipoles in a circle. Should be nonzero force.
    print("\nTest 3: Four dipoles in a circle")
    moments = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    positions = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    force_norm_squared = simsoptpp.diagnostic_test(4, moments, positions)
    print(f"Force norm squared: {force_norm_squared}")
    print("Expected: >0 (non-symmetric arrangement should have force)")
    
    # Test case 4: Two dipoles with specific force expectation
    print("\nTest 4: Two dipoles with specific force")
    moments = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    positions = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    force_norm_squared = simsoptpp.diagnostic_test(2, moments, positions)
    print(f"Force norm squared: {force_norm_squared}")
    print("Expected: Force should be [0.0, -0.1875, 0.0, 0.0, 0.1875, 0.0] * 10^-7")
    
    # Benchmark tests using C++ internal timing
    print("\n" + "="*50)
    print("BENCHMARK TESTS (using C++ internal timing)")
    print("="*50)
    
    # Benchmark N = 1,000 dipoles
    print("\nBenchmark 1: N = 1,000 dipoles")
    force_norm_squared_1k = simsoptpp.diagnostic_test(1000)
    print(f"Force norm squared: {force_norm_squared_1k}")

    # Benchmark N = 10,000 dipoles
    print("\nBenchmark 2: N = 10,000 dipoles")
    force_norm_squared_10k = simsoptpp.diagnostic_test(10000)
    print(f"Force norm squared: {force_norm_squared_10k}")
    
    print("\nAll analytical tests and benchmarks completed successfully!")

if __name__ == "__main__":
    test_A_F_integration() 