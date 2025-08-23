#!/usr/bin/env python3
"""
Script to fix JSON syntax errors in Jupyter notebooks
Specifically handles backslash escaping issues
"""

import json
import re
import sys

def find_json_error(file_path):
    """
    Find the exact location of JSON syntax errors
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse and get detailed error info
        try:
            json.loads(content)
            print("✅ JSON is valid")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON Error at line {e.lineno}, column {e.colno}")
            print(f"Error: {e.msg}")
            
            # Show context around the error
            lines = content.split('\n')
            start_line = max(0, e.lineno - 3)
            end_line = min(len(lines), e.lineno + 3)
            
            print("\nContext around error:")
            for i in range(start_line, end_line):
                marker = ">>> " if i == e.lineno - 1 else "    "
                print(f"{marker}{i+1:3d}: {lines[i]}")
            
            return False
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def fix_common_issues(file_path):
    """
    Fix common JSON issues in Jupyter notebooks
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix 1: Escape backslashes in print statements
        content = re.sub(
            r'"print\(\'\\\\n([^\']*)\'\'\)\\\\n",',
            r'"print(\'\\\\\\\\n\1\')\\\\n",',
            content
        )
        
        # Fix 2: More specific fix for the problematic line
        content = content.replace(
            '"print(\'\\n🎯 Enhanced Highway Guardian training pipeline completed!\')\\n",',
            '"print(\'\\\\n🎯 Enhanced Highway Guardian training pipeline completed!\')\\\\n",'
        )
        
        # Fix 3: Handle other backslash issues
        content = re.sub(
            r'"([^"]*?)\\n([^"]*?)"',
            lambda m: f'"{m.group(1)}\\\\n{m.group(2)}"' if '\\\\n' not in m.group(0) else m.group(0),
            content
        )
        
        if content != original_content:
            # Create backup
            backup_path = file_path.replace('.ipynb', '_backup.ipynb')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Applied fixes to {file_path}")
            print(f"📁 Backup saved as {backup_path}")
            return True
        else:
            print("ℹ️ No fixes needed")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_json_syntax.py <notebook_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("🔍 Analyzing JSON structure...")
    if not find_json_error(file_path):
        print("\n🔧 Attempting to fix common issues...")
        if fix_common_issues(file_path):
            print("\n🔍 Re-checking after fixes...")
            success = find_json_error(file_path)
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)
    else:
        sys.exit(0)