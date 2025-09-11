#!/usr/bin/env python3
"""
Parallelized Caesar Cipher Secure Word Set Finder with 2D Matrix Visualization
Efficiently finds sets of real English words that can all encrypt to the same ciphertext
using different keys, demonstrating perfect secrecy.
"""

import urllib.request
import os
import itertools
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from typing import List, Dict, Set, Tuple, Optional

# Caesar cipher setup
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
CHAR_TO_NUM = {char: i for i, char in enumerate(ALPHABET)}
NUM_TO_CHAR = list(ALPHABET)

def caesar_cipher(message: str, key: str) -> str:
    shift = CHAR_TO_NUM[key]
    encrypted_message = ""
    
    for char in message:
        if char not in CHAR_TO_NUM:
            raise ValueError(f"Character '{char}' in message is not in the alphabet")
        
        char_num = CHAR_TO_NUM[char]
        encrypted_num = (char_num + shift) % 27
        encrypted_message += NUM_TO_CHAR[encrypted_num]
    
    return encrypted_message

def find_key_for_encryption(message: str, target: str) -> Optional[str]:
    if len(message) != len(target):
        return None
    
    shift = (CHAR_TO_NUM[target[0]] - CHAR_TO_NUM[message[0]]) % 27
    
    for i in range(len(message)):
        expected_shift = (CHAR_TO_NUM[target[i]] - CHAR_TO_NUM[message[i]]) % 27
        if expected_shift != shift:
            return None
    
    return NUM_TO_CHAR[shift]

def download_dictionary(cache_file="english_dictionary.txt", force_download=False, max_words=10000):
    if os.path.exists(cache_file) and not force_download:
        print(f"Loading cached dictionary from {cache_file}...")
        with open(cache_file, 'r') as f:
            words = [line.strip() for line in f]
        print(f"Loaded {len(words)} words from cache")
        if max_words and len(words) > max_words:
            words = words[:max_words]
            print(f"Limited to {max_words} words for performance")
        return words
    
    print("Downloading comprehensive English dictionary...")
    all_words = set()
    
    sources = [
        {
            'name': 'MIT Common Words (10k)',
            'url': 'https://www.mit.edu/~ecprice/wordlist.10000'
        },
        {
            'name': 'Popular Words',
            'url': 'https://raw.githubusercontent.com/dolph/dictionary/master/popular.txt'
        }
    ]
    
    for source in sources:
        try:
            print(f"  Downloading {source['name']}...")
            with urllib.request.urlopen(source['url']) as response:
                content = response.read().decode('utf-8')
                words = content.strip().split('\n')
                words = [w.strip().upper() for w in words]
                words = [w for w in words if w and all(c in ALPHABET for c in w)]
                before = len(all_words)
                all_words.update(words)
                added = len(all_words) - before
                print(f"    Added {added} new words")
        except Exception as e:
            print(f"    Failed to download: {e}")
    
    final_words = sorted(list(all_words))
    
    if max_words and len(final_words) > max_words:
        final_words = final_words[:max_words]
        print(f"Limited to {max_words} words for performance")
    
    with open(cache_file, 'w') as f:
        for word in final_words:
            f.write(word + '\n')
    
    print(f"\nTotal unique words: {len(final_words)}")
    print(f"Dictionary cached to {cache_file}")
    
    return final_words

def process_word_batch(word_batch: List[str]) -> Dict[str, List[str]]:
    result = {}
    for word in word_batch:
        ciphertexts = []
        for key in ALPHABET:
            ct = caesar_cipher(word, key)
            ciphertexts.append(ct)
        result[word] = ciphertexts
    return result

def create_ciphertext_matrix_parallel(words: List[str], num_processes: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    if num_processes is None:
        num_processes = min(cpu_count(), 8)
    
    print(f"\nCreating ciphertext matrix using {num_processes} processes...")
    start_time = time.time()
    
    # Split words into batches for parallel processing
    batch_size = max(1, len(words) // (num_processes * 4))
    word_batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
    
    # Process batches in parallel
    with Pool(num_processes) as pool:
        batch_results = pool.map(process_word_batch, word_batches)
    
    # Combine results
    all_ciphertexts = {}
    for batch_result in batch_results:
        all_ciphertexts.update(batch_result)
    
    # Create DataFrame
    matrix_data = []
    for word in words:
        if word in all_ciphertexts:
            row = {'word': word}
            for i, key in enumerate(ALPHABET):
                key_display = f"key_{key}" if key != ' ' else "key_SPACE"
                row[key_display] = all_ciphertexts[word][i]
            matrix_data.append(row)
    
    df = pd.DataFrame(matrix_data)
    df.set_index('word', inplace=True)
    
    # Create reverse mapping (ciphertext -> words that can produce it)
    ct_to_words = defaultdict(set)
    for word, ciphertexts in all_ciphertexts.items():
        for ct in ciphertexts:
            ct_to_words[ct].add(word)
    
    elapsed = time.time() - start_time
    print(f"Matrix created in {elapsed:.2f} seconds")
    print(f"Matrix shape: {df.shape}")
    
    return df, dict(ct_to_words)

def find_ciphertext_collisions_parallel(ct_to_words: Dict[str, Set[str]], 
                                       min_words: int = 2) -> List[Dict]:
    """Find all ciphertexts that can be produced by multiple words."""
    collisions = []
    
    for ct, words in ct_to_words.items():
        if len(words) >= min_words:
            # Find which keys each word needs to produce this ciphertext
            word_key_pairs = []
            for word in words:
                key = find_key_for_encryption(word, ct)
                if key:
                    word_key_pairs.append((word, key))
            
            if len(word_key_pairs) >= min_words:
                collisions.append({
                    'ciphertext': ct,
                    'words': [w for w, _ in word_key_pairs],
                    'keys': {w: k for w, k in word_key_pairs},
                    'count': len(word_key_pairs)
                })
    
    # Sort by number of words that can produce each ciphertext
    collisions.sort(key=lambda x: x['count'], reverse=True)
    
    return collisions

def analyze_word_length_groups(words: List[str], 
                              target_lengths: List[int] = [3, 4, 5],
                              num_processes: Optional[int] = None) -> Dict[int, Dict]:
    """Analyze words grouped by length with parallel processing."""
    results = {}
    
    for length in target_lengths:
        length_words = [w for w in words if len(w) == length]
        
        if not length_words:
            print(f"\nNo {length}-letter words found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {len(length_words)} words of length {length}")
        print(f"{'='*60}")
        
        # Create matrix for this length group
        df_matrix, ct_to_words = create_ciphertext_matrix_parallel(length_words, num_processes)
        
        # Find collisions
        collisions = find_ciphertext_collisions_parallel(ct_to_words, min_words=2)
        
        # Find secure sets of size 3 or more
        secure_sets = []
        seen_sets = set()
        
        for collision in collisions[:100]:  # Analyze top 100 collisions
            if collision['count'] >= 3:
                # Generate combinations of 3 words
                for combo in itertools.combinations(sorted(collision['words']), 3):
                    combo_key = tuple(sorted(combo))
                    if combo_key not in seen_sets:
                        seen_sets.add(combo_key)
                        secure_sets.append({
                            'words': list(combo),
                            'ciphertext': collision['ciphertext'],
                            'keys': {w: collision['keys'][w] for w in combo}
                        })
                        
                        if len(secure_sets) >= 20:  # Limit per length
                            break
            
            if len(secure_sets) >= 20:
                break
        
        results[length] = {
            'matrix': df_matrix,
            'collisions': collisions,
            'secure_sets': secure_sets,
            'total_words': len(length_words),
            'unique_ciphertexts': len(ct_to_words),
            'collision_count': len(collisions)
        }
        
        print(f"  Total unique ciphertexts: {len(ct_to_words)}")
        print(f"  Ciphertexts with collisions: {len(collisions)}")
        print(f"  Secure 3-word sets found: {len(secure_sets)}")
    
    return results

def save_matrix_to_csv(df: pd.DataFrame, filename: str):
    """Save the ciphertext matrix to CSV file."""
    df.to_csv(filename)
    print(f"  Matrix saved to {filename}")

def create_visualization_report(results: Dict[int, Dict], output_dir: str = "caesar_analysis"):
    """Create comprehensive visualization and report files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save matrices for each word length
    for length, data in results.items():
        matrix_file = os.path.join(output_dir, f"matrix_{length}_letters.csv")
        save_matrix_to_csv(data['matrix'], matrix_file)
    
    # Create collision heatmap data
    for length, data in results.items():
        collision_data = []
        for collision in data['collisions'][:50]:  # Top 50 collisions
            collision_data.append({
                'ciphertext': collision['ciphertext'],
                'word_count': collision['count'],
                'words': ', '.join(collision['words'][:5])  # Show first 5 words
            })
        
        if collision_data:
            collision_df = pd.DataFrame(collision_data)
            collision_file = os.path.join(output_dir, f"collisions_{length}_letters.csv")
            collision_df.to_csv(collision_file, index=False)
            print(f"  Collision analysis saved to {collision_file}")
    
    # Create main report
    report_file = os.path.join(output_dir, "analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write("CAESAR CIPHER PARALLEL ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        for length, data in results.items():
            f.write(f"\n{length}-LETTER WORDS ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total words analyzed: {data['total_words']}\n")
            f.write(f"Unique ciphertexts possible: {data['unique_ciphertexts']}\n")
            f.write(f"Ciphertexts with collisions: {data['collision_count']}\n")
            f.write(f"Secure 3-word sets found: {len(data['secure_sets'])}\n\n")
            
            if data['secure_sets']:
                f.write("TOP SECURE WORD SETS:\n")
                for i, secure_set in enumerate(data['secure_sets'][:10], 1):
                    f.write(f"\n  {i}. {secure_set['words']}\n")
                    f.write(f"     Common ciphertext: '{secure_set['ciphertext']}'\n")
                    for word, key in secure_set['keys'].items():
                        shift = CHAR_TO_NUM[key]
                        key_display = "SPACE" if key == " " else key
                        f.write(f"     '{word}' + key '{key_display}' (shift {shift:2}) → '{secure_set['ciphertext']}'\n")
            
            f.write("\n" + "=" * 70 + "\n")
    
    print(f"\nAnalysis report saved to {report_file}")
    
    # Save JSON with all secure sets
    json_file = os.path.join(output_dir, "secure_sets.json")
    all_secure_sets = []
    for length, data in results.items():
        for secure_set in data['secure_sets']:
            all_secure_sets.append({
                'word_length': length,
                **secure_set
            })
    
    with open(json_file, 'w') as f:
        json.dump({
            'description': 'Semantically secure word sets found through parallel analysis',
            'total_sets': len(all_secure_sets),
            'sets': all_secure_sets
        }, f, indent=2)
    
    print(f"Secure sets JSON saved to {json_file}")


def main():
    print("=" * 70)
    print("PARALLELIZED CAESAR CIPHER ANALYZER WITH 2D MATRIX")
    print("=" * 70)
    
    # Configuration
    NUM_PROCESSES = min(cpu_count(), 8)
    MAX_WORDS = 370000  # Limit for performance (set to None for all words)
    TARGET_LENGTHS = [3, 4, 5,6,7,8,9,10]  # Word lengths to analyze
    
    print(f"\nConfiguration:")
    print(f"  Processes: {NUM_PROCESSES}")
    print(f"  Max words: {MAX_WORDS if MAX_WORDS else 'Unlimited'}")
    print(f"  Target lengths: {TARGET_LENGTHS}")
    
    # Download/load dictionary
    words = download_dictionary(max_words=MAX_WORDS)
    
    # Perform parallel analysis
    print("\n" + "=" * 70)
    print("PERFORMING PARALLEL ANALYSIS")
    print("=" * 70)
    
    start_time = time.time()
    results = analyze_word_length_groups(words, TARGET_LENGTHS, NUM_PROCESSES)
    total_time = time.time() - start_time
    
    print(f"\nTotal analysis time: {total_time:.2f} seconds")
    
    # Find interesting patterns
    print("\n" + "=" * 70)
    print("FINDING SEMANTICALLY INTERESTING PATTERNS")
    print("=" * 70)
    
    # Create visualization files
    print("\n" + "=" * 70)
    print("CREATING OUTPUT FILES")
    print("=" * 70)
    
    create_visualization_report(results)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    total_secure_sets = sum(len(data['secure_sets']) for data in results.values())
    total_collisions = sum(data['collision_count'] for data in results.values())
    
    print(f"\nTotal secure 3-word sets found: {total_secure_sets}")
    print(f"Total ciphertext collisions found: {total_collisions}")

if __name__ == "__main__":
    main()