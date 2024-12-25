import numpy as np

def circular_shift(code, shift):
    """
    Circularly shift a binary code (list of 0s and 1s).
    
    Args:
        code: List of binary values (e.g., [1, 0, 1, 0, 1]).
        shift: Number of positions to shift. Positive for left, negative for right.
    
    Returns:
        Shifted binary code.
    """
    n = len(code)
    shift = shift % n  # Handle shifts greater than the length of the code
    return code[shift:] + code[:shift]

def HammingDistance(code1, code2, mask1=None, mask2=None):
    """
    Compute the Hamming distance between two iris codes.
    :param code1: First binary iris code (numpy array).
    :param code2: Second binary iris code (numpy array).
    :param mask1: Mask for code1 (optional).
    :param mask2: Mask for code2 (optional).
    :return: Hamming distance.
    """
    if mask1 is not None and mask2 is not None:
        # Combine masks and exclude masked bits
        combined_mask = mask1 & mask2
        differing_bits = np.sum((code1 ^ code2) & ~combined_mask)
        total_bits = np.sum(~combined_mask)
    else:
        # Without masks
        differing_bits = np.sum(code1 ^ code2)
        total_bits = code1.size
    
    if total_bits == 0:
        return float('inf')  # Handle case where no bits are available for comparison
    return differing_bits / total_bits

def find_min_hamming_distance(reference_code, query_code):
    """
    Find the minimum Hamming distance by circularly shifting the query code.
    
    Args:
        reference_code: The reference binary code (list of 0s and 1s).
        query_code: The query binary code (list of 0s and 1s).
    
    Returns:
        Tuple containing the minimum Hamming distance and the optimal shift value.
    """
    n = len(query_code)
    min_distance = float('inf')
    best_shift = 0
    
    for shift in range(n):
        shifted_code = circular_shift(query_code, shift)
        reference_code = np.array(reference_code)
        shifted_code = np.array(shifted_code)
        distance = HammingDistance(reference_code, shifted_code)
        if distance < min_distance:
            min_distance = distance
            best_shift = shift
    
    return min_distance, best_shift