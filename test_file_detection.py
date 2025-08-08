#!/usr/bin/env python3
"""
Test script to demonstrate the new file detection capabilities
"""

import sys
from pathlib import Path

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.file_detector import get_csv_files_with_metadata
from src.config import get_config

def main():
    print("🎯 Testing New Config-Based File Detection")
    print("=" * 50)
    
    # Load configuration
    config = get_config()
    file_config = config.get_file_handling_config()
    
    print("📁 Configured directories:")
    for i, directory in enumerate(file_config['default_directories'], 1):
        exists = "✅" if Path(directory).exists() else "❌"
        print(f"  {i}. {exists} {directory}")
    
    print(f"\n🔍 File patterns:")
    for pattern in file_config['file_patterns']:
        print(f"  • {pattern}")
    
    print(f"\n📊 Settings:")
    print(f"  • Max files shown: {file_config['max_files_shown']}")
    print(f"  • Sort by: {file_config['sort_by']}")
    
    print(f"\n🚀 Searching for transaction files...")
    files = get_csv_files_with_metadata(max_files=10)
    
    if not files:
        print("❌ No files found matching your patterns")
        return
    
    print(f"\n📂 Found {len(files)} matching files:")
    print()
    
    # Group by directory
    files_by_dir = {}
    for file_info in files:
        dir_name = file_info['directory']
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append(file_info)
    
    for directory, dir_files in files_by_dir.items():
        dir_display = directory.replace(str(Path.home()), "~")
        print(f"📁 {dir_display} ({len(dir_files)} files)")
        
        for file_info in dir_files:
            status_icon = "✅" if file_info['is_valid'] else "⚠️"
            source_text = file_info['source'].replace('_', ' ').title()
            
            print(f"  {status_icon} {file_info['name']}")
            print(f"      🎯 {source_text} ({file_info['confidence']} confidence)")
            print(f"      📊 {file_info['row_count']} rows, {file_info['size_human']}")
            print(f"      📅 {file_info['modified']}")
            print()
    
    print("✅ File detection working perfectly!")
    print("💡 Your files are found automatically without moving them to project folder")

if __name__ == "__main__":
    main()