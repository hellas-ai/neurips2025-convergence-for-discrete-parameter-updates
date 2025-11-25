#!/usr/bin/env python3
"""
Script to create a software artifact ZIP file for submission.
Includes only source code and configuration needed to run experiments.
"""

import os
import zipfile
from datetime import datetime
from pathlib import Path

def get_git_commit_hash():
    """Get the current git commit hash for naming."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:8]
    except:
        return "unknown"

def create_artifact_zip():
    """Create a ZIP file containing the software artifact."""
    
    # Get commit hash and timestamp for unique naming
    commit_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"multinomial_sampling_artifact_{commit_hash}_{timestamp}.zip"
    
    print(f"Creating artifact: {zip_name}")
    
    # Files and directories to include
    files_to_include = [
        # Core source code
        "main.py",
        "experiment.py", 
        "optimize.py",
        "plot.py",
        
        # Model definitions
        "models/linear.py",
        "models/hidden.py", 
        "models/convolutional.py",
        "models/resnet.py",
        
        # Configuration and dependencies
        "requirements.txt",
        
        # Documentation
        "README.md",
        
        # Shell scripts for running experiments
        "run_mnist_experiments.sh",
        "run_mnist_model_experiments.sh"
    ]
    
    # Optional files to include if they exist
    optional_files = [
        "PARTIAL_RESULTS.md",
        "TODO.md"
    ]
    
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Add required files
        for item in files_to_include:
            if os.path.isfile(item):
                print(f"Adding file: {item}")
                zipf.write(item)
            elif os.path.isdir(item):
                print(f"Adding directory: {item}")
                for root, dirs, files in os.walk(item):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path)
            else:
                print(f"Warning: {item} not found, skipping")
    
    # Get ZIP file size
    zip_size = os.path.getsize(zip_name)
    zip_size_mb = zip_size / (1024 * 1024)
    
    print(f"\nArtifact created successfully!")
    print(f"File: {zip_name}")
    print(f"Size: {zip_size_mb:.2f} MB")
    print(f"\nContents summary:")
    
    # List contents of the ZIP
    with zipfile.ZipFile(zip_name, 'r') as zipf:
        file_count = len(zipf.namelist())
        print(f"- {file_count} files total")
        
        # Count by category
        py_files = [f for f in zipf.namelist() if f.endswith('.py')]
        
        print(f"- {len(py_files)} Python source files")
        print(f"- Source code only (no experiment data)")
        
        # Show all files
        print(f"\nFiles included:")
        for name in sorted(zipf.namelist()):
            print(f"  {name}")
    
    return zip_name

if __name__ == "__main__":
    artifact_name = create_artifact_zip()
    print(f"\nTo submit: upload {artifact_name}")
