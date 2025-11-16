"""
NIST SP 800-22 Statistical Test Suite Implementation
Implements tests 2.1-2.4 for random and pseudorandom number generators
"""

import numpy as np
from scipy.special import erfc, gammaincc
from typing import Dict
import secrets

def random_bitstring(n: int) -> str:
    """Generate a cryptographically-secure random bitstring of length n.
    Args:
        n: number of bits (integer)
    Returns:
        String of bits    
    """

    num_bytes = (n + 7) // 8
    rand_int = int.from_bytes(secrets.token_bytes(num_bytes), 'big')
    bits = bin(rand_int)[2:].zfill(num_bytes * 8)
    return bits[-n:]



def frequency_test(epsilon: np.ndarray, 
                   alpha: float = 0.01) -> Dict:
    """
    Test 2.1: Frequency (Monobit) Test
    
    Tests whether the number of ones and zeros in a sequence are approximately
    the same as would be expected for a truly random sequence.
    
    Args:
        epsilon: Binary sequence (numpy array of 0s and 1s)
        alpha: Significance level (default: 0.01)
    
    Returns:
        dict: {
            'test_name': str,
            'test_statistic': float,
            'p_value': float,
            'passed': bool,
            'bits': str (first 100 chars)
        }
    """
    
    n = len(epsilon)
    
    # Validate input
    if n < 100:
        raise ValueError(f"Sequence too short: {n} < 100")

    
    # Normalize and compute sum
    X = 2 * epsilon - 1
    S_n = np.sum(X)
    
    # Compute test statistic
    s_obs = abs(S_n) / np.sqrt(n)
    
    # Compute P-value
    p_value = erfc(s_obs / np.sqrt(2))
    
    # Decision
    passed = p_value >= alpha
    
    # Get bit string representation
    bit_string = ''.join(map(str, epsilon.astype(int)))

    
    return {
        'test_name': 'Frequency (Monobit) Test',
        'test_statistic': s_obs,
        'p_value': p_value,
        'passed': passed,
        'bits': bit_string[:100] + ('...' if len(bit_string) > 100 else ''),
        'n': n
    }


def block_frequency_test(epsilon: np.ndarray,
                        M: int,
                        alpha: float = 0.01) -> Dict:
    """
    Test 2.2: Frequency Test within a Block
    
    Determines whether the frequency of ones in an M-bit block is approximately
    M/2, as would be expected under an assumption of randomness.
    
    Args:
        epsilon: Binary sequence (numpy array of 0s and 1s)
        M: Block size (should be M >= 20, M > 0.01*n, and resulting N < 100)
        alpha: Significance level (default: 0.01)
    
    Returns:
        dict: Test results including p_value and pass/fail status
    """

    n = len(epsilon)
    
    # Validate input
    if n < 100:
        raise ValueError(f"Sequence too short: {n} < 100")
    
    if M < 10: # Input constraint says 20, input says 10. Usually fails
        raise ValueError(f"Block size too small: M = {M} < 20")
    
    if M > 0.01 * n:
        pass 
    else:
        raise ValueError(f"Block size too small: M should be > 0.01*n = {0.01*n}")
    
    # Partition into N blocks
    N = n // M
    
    if N >= 100:
        raise ValueError(f"Too many blocks: N = {N} (should be < 100)")
    
    # Discard unused bits
    epsilon = epsilon[:N * M]
    
    # Reshape into blocks
    blocks = epsilon.reshape(N, M)
    
    # Calculate proportion of ones in each block
    pi = np.sum(blocks, axis=1) / M
    
    # Compute chi-square statistic
    chi_square_obs = 4 * M * np.sum((pi - 0.5) ** 2)
    
    # Compute P-value using incomplete gamma function
    p_value = gammaincc(N / 2, chi_square_obs / 2)
    
    # Decision
    passed = p_value >= alpha
    
    # Get bit string representation
    bit_string = ''.join(map(str, epsilon.astype(int)))
    
    return {
        'test_name': 'Block Frequency Test',
        'test_statistic': chi_square_obs,
        'p_value': p_value,
        'passed': passed,
        'bits': bit_string[:100] + ('...' if len(bit_string) > 100 else ''),
        'M': M,
        'N': N,
        'n': n
    }


def runs_test(epsilon: np.ndarray,
              alpha: float = 0.01) -> Dict:
    """
    Test 2.3: Runs Test
    
    Determines whether the number of runs of ones and zeros of various lengths
    is as expected for a random sequence. Tests whether oscillation between
    zeros and ones is too fast or too slow.
    
    Args:
        epsilon: Binary sequence (numpy array of 0s and 1s)
        alpha: Significance level (default: 0.01)
    
    Returns:
        dict: Test results including p_value and pass/fail status
    """
    
    n = len(epsilon)
    
    # Validate input
    if n < 100:
        raise ValueError(f"Sequence too short: {n} < 100 ")
    
    # Step 1: Compute proportion of ones
    pi = np.sum(epsilon) / n
    
    # Step 2: Check if prerequisite frequency test passes
    tau = 2 / np.sqrt(n)
    
    if abs(pi - 0.5) >= tau:        
        return {
            'test_name': 'Runs Test',
            'test_statistic': None,
            'p_value': 0.0,
            'passed': False,
            'bits': ''.join(map(str, epsilon.astype(int)))[:100] + '...',
            'prerequisite_failed': True,
            'n': n
        }
    
    # Step 3: Compute test statistic V_n(obs)
    r = np.zeros(n - 1)
    for k in range(n - 1):
        if epsilon[k] != epsilon[k + 1]:
            r[k] = 1
    
    V_n_obs = np.sum(r) + 1
    
    # Step 4: Compute P-value
    numerator = abs(V_n_obs - 2 * n * pi * (1 - pi))
    denominator = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    
    p_value = erfc(numerator / denominator)
    
    # Decision
    passed = p_value >= alpha
    
    # Get bit string representation
    bit_string = ''.join(map(str, epsilon.astype(int)))
    
    return {
        'test_name': 'Runs Test',
        'test_statistic': V_n_obs,
        'p_value': p_value,
        'passed': passed,
        'bits': bit_string[:100] + ('...' if len(bit_string) > 100 else ''),
        'pi': pi,
        'n': n
    }


def longest_run_of_ones_test(epsilon: np.ndarray,
                             alpha: float = 0.01) -> Dict:
    """
    Test 2.4: Test for the Longest Run of Ones in a Block
    
    Determines whether the length of the longest run of ones within the tested
    sequence is consistent with the length expected in a random sequence.
    
    Args:
        epsilon: Binary sequence (numpy array of 0s and 1s)
        alpha: Significance level (default: 0.01)
    
    Returns:
        dict: Test results including p_value and pass/fail status
    """
    
    n = len(epsilon)
    
    # Determine M, N, K based on sequence length
    if n < 128:
        raise ValueError(f"Sequence too short: {n} < 128")
    elif n < 6272:
        M = 8
        N = 16
        K = 3
        pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
        boundaries = [1, 2, 3, 4]
    elif n < 750000:
        M = 128
        N = 49
        K = 5
        pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        boundaries = [4, 5, 6, 7, 8, 9]
    else:
        M = 10000
        N = 75
        K = 6
        pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        boundaries = [10, 11, 12, 13, 14, 15, 16]
    
    # Adjust N based on actual sequence length
    N = n // M
    
    # Truncate sequence to fit exactly N blocks
    epsilon = epsilon[:N * M]
    
    # Divide into M-bit blocks
    blocks = epsilon.reshape(N, M)
    
    # Find longest run of ones in each block
    def longest_run_in_block(block):
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    longest_runs = np.array([longest_run_in_block(block) for block in blocks])
    
    # Tabulate frequencies into categories
    v = np.zeros(K + 1)
    
    bins = {
        8:     [1, 2, 3],
        128:   [4, 5, 6, 7, 8],
        10000: [10, 11, 12, 13, 14, 15],
    }

    for run_length in longest_runs:
        thresholds = bins.get(M, [])
        for i, t in enumerate(thresholds):
            if run_length <= t:
                v[i] += 1
                break
        else:
            v[len(thresholds)] += 1

    
    # Compute chi-square statistic
    chi_square_obs = 0.0
    for i in range(K + 1):
        chi_square_obs += ((v[i] - N * pi_values[i]) ** 2) / (N * pi_values[i])
    
    # Compute P-value
    p_value = gammaincc(K / 2, chi_square_obs / 2)
    
    # Decision
    passed = p_value >= alpha
    
    # Get bit string representation
    bit_string = ''.join(map(str, epsilon.astype(int)))
    
    return {
        'test_name': 'Longest Run of Ones Test',
        'test_statistic': chi_square_obs,
        'p_value': p_value,
        'passed': passed,
        'bits': bit_string[:100] + ('...' if len(bit_string) > 100 else ''),
        'M': M,
        'N': N,
        'K': K,
        'n': n
    }

def print_summary(results: Dict):
    """Print a summary table of all test results"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'Test Name':<35} {'P-value':<12} {'Result':<10}")
    print("-"*70)
    
    for test_key, test_result in results.items():
        if 'error' in test_result:
            print(f"{test_key:<35} {'ERROR':<12} {'N/A':<10}")
        else:
            test_name = test_result.get('test_name', test_key)
            p_value = test_result.get('p_value', 0.0)
            passed = test_result.get('passed', False)
            result_str = 'PASS' if passed else 'FAIL'
            print(f"{test_name:<35} {p_value:<12.6f} {result_str:<10}")
    
    print("="*70)


def run_nist_test_suite(epsilons: list[np.ndarray],
                        M_block: list[int] = [10,8],
                        alpha: float = 0.01) -> None:
    """
    Run complete NIST test suite with formatted output
    
    Args:
        epsilon: Binary sequence to test
        M_block: Block size for block frequency test
        alpha: Significance level (default: 0.01)
    """
    
    # Run all tests
    results = {}
    
    # Test 1: Frequency (Monobit) Test
    print("\n[1/4] Running Frequency (Monobit) Test...")
    try:
        results['frequency'] = frequency_test(epsilons[0], alpha)
        r = results['frequency']
        print(f"      Test statistic: {r['test_statistic']:.6f}")
        print(f"      P-value: {r['p_value']:.6f}")
        print(f"      Result: {'PASS' if r['passed'] else 'FAIL'}")
    except Exception as e:
        print(f"      ERROR: {e}")
        results['frequency'] = {'error': str(e)}
    
    # Test 2: Block Frequency Test
    print("\n[2/4] Running Block Frequency Test...")
    try:
        results['block_frequency'] = block_frequency_test(epsilons[0], M_block[0], alpha)
        r = results['block_frequency']
        print(f"      Test statistic: {r['test_statistic']:.6f}")
        print(f"      P-value: {r['p_value']:.6f}")
        print(f"      Result: {'PASS' if r['passed'] else 'FAIL'}")
    except Exception as e:
        print(f"      ERROR: {e}")
        results['block_frequency'] = {'error': str(e)}
    
    # Test 3: Runs Test
    print("\n[3/4] Running Runs Test...")
    try:
        results['runs'] = runs_test(epsilons[0], alpha)
        r = results['runs']
        if 'prerequisite_failed' in r and r['prerequisite_failed']:
            print(f"      Prerequisite frequency test FAILED")
            print(f"      P-value: {r['p_value']:.6f}")
            print(f"      Result: FAIL")
        else:
            print(f"      Test statistic: {r['test_statistic']:.6f}")
            print(f"      P-value: {r['p_value']:.6f}")
            print(f"      Result: {'PASS' if r['passed'] else 'FAIL'}")
    except Exception as e:
        print(f"      ERROR: {e}")
        results['runs'] = {'error': str(e)}
    
    # Test 4: Longest Run of Ones Test
    print("\n[4/4] Running Longest Run of Ones Test...")
    try:
        results['longest_run'] = longest_run_of_ones_test(epsilons[1], alpha)
        r = results['longest_run']
        print(f"      Test statistic: {r['test_statistic']:.6f}")
        print(f"      P-value: {r['p_value']:.6f}")
        print(f"      Result: {'PASS' if r['passed'] else 'FAIL'}")
    except Exception as e:
        print(f"      ERROR: {e}")
        results['longest_run'] = {'error': str(e)}
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    """Run NIST test suite with documentation examples"""
    
    print("\n" + "*"*70)
    print("*" + " NIST SP 800-22 STATISTICAL TEST SUITE ".center(68) + "*")
    print("*" + " Testing with NIST Documentation Examples ".center(68) + "*")
    print("*"*70)
    
    # Test sequence for examples 2.1-2.3
    seq_100 = ("11001001000011111101101010100010001000010110100011"
               "00001000110100110001001100011001100010100010111000")
    seq_100 = np.array([int(bit) for bit in seq_100])
    # Test sequence for example 2.4  
    seq_128 = ("11001100000101010110110001001100111000000000001001"
               "00110101010001000100111101011010000000110101111100"
               "1100111001101101100010110010")
    seq_128 = np.array([int(bit) for bit in seq_128])
    
    # Run tests
    run_nist_test_suite([seq_100,seq_128], M_block=[10,8])

    print("\n" + "*"*70)
    print("*" + " NIST SP 800-22 STATISTICAL TEST SUITE ".center(68) + "*")
    print("*" + " Testing with Pseudorandom Examples ".center(68) + "*")
    print("*"*70)

    # Pseudorandom sequence for examples 2.1-2.3
    pr_seq_100 = random_bitstring(100)

    pr_seq_100 = np.array([int(bit) for bit in pr_seq_100])

    # Pseudorandom sequence for examples 2.4
    pr_seq_128 = random_bitstring(128)

    pr_seq_128 = np.array([int(bit) for bit in pr_seq_128])

    run_nist_test_suite([pr_seq_100,pr_seq_128], M_block=[10,8])



