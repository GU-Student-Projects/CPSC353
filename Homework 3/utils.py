#!/usr/bin/env python3
"""
Utilities for Vigenère cipher analysis including dictionary management,
frequency analysis helpers, and web search functions.
"""

import urllib.request
import os
import requests

def add_space_and_rebalance(original_freq, space_multiplier=2.0):
    """
    Adds space and rebalances frequencies to sum to 1.0
    
    Parameters:
    - original_freq: Dictionary of character frequencies (A-Z)
    - space_multiplier: How many times more frequent space is than 'E'
    
    Returns:
    - Balanced dictionary with space added and all frequencies summing to 1.0
    """
    # First, normalize original frequencies to sum to exactly 1.0
    original_sum = sum(original_freq.values())
    normalized_freq = {char: freq/original_sum for char, freq in original_freq.items()}
    
    # Get normalized E frequency
    e_freq_normalized = normalized_freq['E']
    
    # Calculate what space frequency should be
    space_freq = e_freq_normalized * space_multiplier
    
    # Calculate scale factor for other characters
    scale_factor = 1 - space_freq
    
    # Create new dictionary with scaled frequencies
    balanced_freq = {}
    balanced_freq[' '] = space_freq
    
    for char, freq in normalized_freq.items():
        balanced_freq[char] = freq * scale_factor
    
    # Verify
    total = sum(balanced_freq.values())
    print(f"Total frequency sum: {total:.10f}")
    
    return balanced_freq

def download_dictionary(cache_file="english_dictionary.txt", 
                       force_download=False, 
                       max_words=None,
                       min_word_length=2,
                       max_word_length=None):
    """
    Download and cache a comprehensive English dictionary.
    
    Parameters:
    - cache_file: Path to cache file
    - force_download: Force re-download even if cache exists
    - max_words: Maximum number of words to load (None for all)
    - min_word_length: Minimum word length to include
    - max_word_length: Maximum word length to include (None for no limit)
    
    Returns:
    - Set of uppercase English words
    """
    # Check cache first
    if os.path.exists(cache_file) and not force_download:
        print(f"Loading cached dictionary from {cache_file}...")
        with open(cache_file, 'r') as f:
            words = set()
            for line in f:
                word = line.strip().upper()
                if min_word_length <= len(word):
                    if max_word_length is None or len(word) <= max_word_length:
                        words.add(word)
                        if max_words and len(words) >= max_words:
                            break
        
        print(f"Loaded {len(words)} words from cache")
        return words
    
    print("Downloading comprehensive English dictionary...")
    all_words = set()
    
    # Dictionary sources
    sources = [
        {
            'name': 'MIT Common Words (10k)',
            'url': 'https://www.mit.edu/~ecprice/wordlist.10000'
        },
        {
            'name': 'Popular English Words',
            'url': 'https://raw.githubusercontent.com/dolph/dictionary/master/popular.txt'
        },
        {
            'name': 'English Word List',
            'url': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt'
        }
    ]
    
    for source in sources:
        try:
            print(f"  Downloading {source['name']}...")
            with urllib.request.urlopen(source['url'], timeout=10) as response:
                content = response.read().decode('utf-8')
                words = content.strip().split('\n')
                
                # Process and filter words
                valid_words = []
                for w in words:
                    w = w.strip().upper()
                    # Only keep alphabetic words
                    if w and all(c.isalpha() for c in w):
                        if min_word_length <= len(w):
                            if max_word_length is None or len(w) <= max_word_length:
                                valid_words.append(w)
                
                before = len(all_words)
                all_words.update(valid_words)
                added = len(all_words) - before
                print(f"    Added {added} new words (total: {len(all_words)})")
                
        except Exception as e:
            print(f"    Failed to download {source['name']}: {e}")
            continue
    
    # Sort and limit if necessary
    final_words = sorted(list(all_words))
    
    if max_words and len(final_words) > max_words:
        final_words = final_words[:max_words]
        print(f"Limited to {max_words} words")
    
    # Cache the results
    with open(cache_file, 'w') as f:
        for word in final_words:
            f.write(word + '\n')
    
    print(f"\nTotal unique words: {len(final_words)}")
    print(f"Dictionary cached to {cache_file}")
    
    # Convert to set for faster lookups
    return set(final_words)

def search_google_books(search_text, num_results=3):
    """
    Search Google Books API to identify a text passage.
    
    Parameters:
    - search_text: Text passage to search for
    - num_results: Number of results to analyze
    
    Returns:
    - Dictionary with identified work and author, or None if not found
    """    
    # Clean the search text
    search_text = ' '.join(search_text.split())[:200]  # Normalize spaces, limit length
    
    try:
        # Google Books API endpoint (no API key needed for basic search)
        api_url = "https://www.googleapis.com/books/v1/volumes"
        
        # Try exact phrase search first
        params = {
            'q': f'"{search_text[:150]}"',  # Exact phrase in quotes
            'maxResults': num_results,
            'printType': 'books'
        }
        
        print(f"Searching Google Books API...")
        response = requests.get(api_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # If no results with exact match, try without quotes
            if 'items' not in data or len(data['items']) == 0:
                print("No exact matches, trying broader search...")
                params['q'] = search_text[:150]  # Without quotes
                response = requests.get(api_url, params=params, timeout=10)
                data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                # Analyze results
                results = []
                for item in data['items']:
                    book_info = item.get('volumeInfo', {})
                    
                    title = book_info.get('title', 'Unknown Title')
                    authors = book_info.get('authors', ['Unknown Author'])
                    author = authors[0] if authors else 'Unknown Author'
                    published = book_info.get('publishedDate', 'Unknown')
                    description = book_info.get('description', '')
                    categories = book_info.get('categories', [])
                    preview_link = book_info.get('previewLink', '')
                    
                    # Calculate confidence based on various factors
                    confidence = 0
                    
                    # Check if it's classic literature
                    if any(cat in str(categories).lower() for cat in ['fiction', 'classic', 'literature']):
                        confidence += 30
                    
                    # Check publication date (older books more likely to be classics)
                    try:
                        year = int(published[:4]) if published != 'Unknown' else 2000
                        if year < 1950:
                            confidence += 20
                        elif year < 1980:
                            confidence += 10
                    except:
                        pass
                    
                    # Check for known classic authors
                    classic_authors = ['shakespeare', 'austen', 'dickens', 'twain', 'wilde', 
                                     'bronte', 'tolstoy', 'dostoyevsky', 'melville', 'hawthorne',
                                     'poe', 'carroll', 'stevenson', 'doyle', 'wells', 'verne',
                                     'dumas', 'hugo', 'cervantes', 'homer', 'virgil']
                    
                    author_lower = author.lower()
                    if any(name in author_lower for name in classic_authors):
                        confidence += 40
                    
                    results.append({
                        'title': title,
                        'author': author,
                        'published': published,
                        'categories': categories,
                        'description': description[:200] if description else '',
                        'preview_link': preview_link,
                        'confidence': confidence
                    })
                
                # Sort by confidence
                results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Return the best match
                best_match = results[0]
                
                print(f"\n✓ Found {len(results)} matches in Google Books")
                print(f"Best match: '{best_match['title']}' by {best_match['author']}")
                
                if len(results) > 1:
                    print("\nOther possible matches:")
                    for i, result in enumerate(results[1:3], 1):  # Show next 2
                        print(f"  {i}. '{result['title']}' by {result['author']}")
                
                return {
                    'found': True,
                    'title': best_match['title'],
                    'author': best_match['author'],
                    'published': best_match['published'],
                    'confidence': best_match['confidence'],
                    'all_results': results,
                    'search_query': search_text[:150]
                }
            else:
                print("No results found in Google Books")
                return {
                    'found': False,
                    'search_query': search_text[:150]
                }
                
        else:
            print(f"Google Books API error: Status {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("Google Books API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Google Books API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None