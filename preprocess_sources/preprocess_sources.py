#!/usr/bin/env python3
"""
Script to preprocess C/C++ source files using compile_commands.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional
import shlex


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess C/C++ source files from compile_commands.json'
    )
    parser.add_argument(
        'compile_commands',
        type=str,
        help='Path to compile_commands.json file'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save preprocessed source files'
    )
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=cpu_count(),
        help=f'Number of parallel processes (default: {cpu_count()})'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def load_compile_commands(json_path: str) -> List[Dict]:
    """Load and parse compile_commands.json file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}", file=sys.stderr)
        sys.exit(1)


def is_c_or_cpp_source(filepath: str) -> bool:
    """Check if the file is a C or C++ source file."""
    c_extensions = {'.c'}
    cpp_extensions = {'.cc', '.cpp', '.cxx', '.C', '.c++'}
    ext = Path(filepath).suffix
    return ext in c_extensions or ext in cpp_extensions


def get_preprocessed_extension(filepath: str) -> str:
    """Get the appropriate extension for preprocessed file."""
    c_extensions = {'.c'}
    ext = Path(filepath).suffix
    return '.i' if ext in c_extensions else '.ii'


def generate_preprocess_command(entry: Dict) -> Tuple[str, str, str]:
    """
    Generate preprocessing command from compile command.
    Returns tuple of (command, source_file, output_file)
    """
    command = entry['command']
    source_file = entry['file']
    directory = entry.get('directory', '.')
    
    # Parse the command
    cmd_parts = shlex.split(command)
    
    # Find and remove -o option if present
    new_cmd_parts = []
    skip_next = False
    for i, part in enumerate(cmd_parts):
        if skip_next:
            skip_next = False
            continue
        if part == '-o':
            skip_next = True
            continue
        if part.startswith('-o'):
            continue
        new_cmd_parts.append(part)
    
    # Find where to insert -E flag
    # Look for the actual compiler (gcc, g++, clang, clang++, cc, c++)
    compilers = ['gcc', 'g++', 'clang', 'clang++', 'cc', 'c++']
    insert_index = 1
    for i, part in enumerate(new_cmd_parts):
        if any(part.endswith(comp) for comp in compilers):
            insert_index = i + 1
            break
    
    # Add -E flag for preprocessing only
    if '-E' not in new_cmd_parts:
        new_cmd_parts.insert(insert_index, '-E')
    
    # Remove certain flags that might interfere with preprocessing
    flags_to_remove = ['-c', '-MD', '-MMD', '-MT', '-MF']
    new_cmd_parts = [p for p in new_cmd_parts if not any(p.startswith(f) for f in flags_to_remove)]
    
    # Add -P to omit line markers if desired (optional)
    # new_cmd_parts.insert(insert_index + 1, '-P')
    
    return (new_cmd_parts, source_file, directory)


def preprocess_file(args: Tuple[List[str], str, str, str, bool]) -> Tuple[bool, str, str, int, int]:
    """
    Preprocess a single file.
    Returns tuple of (success, source_file, message, original_size, preprocessed_size)
    """
    cmd_parts, source_file, working_dir, output_path, verbose = args
    
    try:
        # Get original file size
        original_size = 0
        full_source_path = os.path.join(working_dir, source_file) if not os.path.isabs(source_file) else source_file
        if os.path.exists(full_source_path):
            original_size = os.path.getsize(full_source_path)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            print(f"Processing: {source_file}")
            print(f"  Command: {' '.join(cmd_parts)}")
            print(f"  Output: {output_path}")
        
        # Run the preprocessor
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=60  # 60 second timeout per file
        )
        
        if result.returncode != 0:
            return (False, source_file, f"Preprocessing failed: {result.stderr}", original_size, 0)
        
        # Save the preprocessed output
        with open(output_path, 'w') as f:
            f.write(result.stdout)
        
        # Get preprocessed file size
        preprocessed_size = os.path.getsize(output_path)
        
        return (True, source_file, "Success", original_size, preprocessed_size)
        
    except subprocess.TimeoutExpired:
        return (False, source_file, "Timeout during preprocessing", 0, 0)
    except Exception as e:
        return (False, source_file, f"Error: {str(e)}", 0, 0)


def main():
    args = parse_arguments()
    
    # Load compile commands
    print(f"Loading compile commands from: {args.compile_commands}")
    compile_commands = load_compile_commands(args.compile_commands)
    
    # Filter for C/C++ source files
    filtered_commands = [
        entry for entry in compile_commands
        if is_c_or_cpp_source(entry['file'])
    ]
    
    print(f"Found {len(filtered_commands)} C/C++ source files to preprocess")
    
    if not filtered_commands:
        print("No C/C++ source files found in compile_commands.json")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare preprocessing tasks
    tasks = []
    for entry in filtered_commands:
        cmd_parts, source_file, working_dir = generate_preprocess_command(entry)
        
        # Generate output path preserving directory structure
        source_path = Path(source_file)
        if source_path.is_absolute():
            # Remove leading slash for absolute paths
            relative_path = str(source_path).lstrip('/')
        else:
            relative_path = str(source_path)
        
        # Change extension to .i or .ii
        output_filename = source_path.stem + get_preprocessed_extension(str(source_path))
        output_path = output_dir / Path(relative_path).parent / output_filename
        
        tasks.append((cmd_parts, source_file, working_dir, str(output_path), args.verbose))
    
    # Process files in parallel
    print(f"Starting preprocessing with {args.jobs} parallel jobs...")
    
    successful = 0
    failed = 0
    total_original_size = 0
    total_preprocessed_size = 0
    
    with Pool(processes=args.jobs) as pool:
        results = pool.map(preprocess_file, tasks)
        
        for success, source_file, message, original_size, preprocessed_size in results:
            if success:
                successful += 1
                total_original_size += original_size
                total_preprocessed_size += preprocessed_size
                if not args.verbose:
                    print(f"✓ {source_file}")
            else:
                failed += 1
                total_original_size += original_size  # Still count original size for failed files
                print(f"✗ {source_file}: {message}", file=sys.stderr)
    
    # Format file sizes
    def format_size(size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    # Print summary
    print(f"\nPreprocessing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\nFile sizes:")
    print(f"  Original files total: {format_size(total_original_size)} ({total_original_size:,} bytes)")
    print(f"  Preprocessed files total: {format_size(total_preprocessed_size)} ({total_preprocessed_size:,} bytes)")
    if total_original_size > 0:
        expansion_ratio = total_preprocessed_size / total_original_size
        print(f"  Expansion ratio: {expansion_ratio:.2f}x")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()