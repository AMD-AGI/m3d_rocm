#!/usr/bin/env python3
"""
Fix VideoInput import in VideoLLaMA3: move VideoInput to video_utils, keep rest in image_utils
Run this after first model download to fix the HuggingFace cached model.
"""

import os
import re
from pathlib import Path


def ensure_transformers_videoinput_compat(verbose=True):
    """
    Add a compatibility alias for transformers versions that moved VideoInput.
    """
    try:
        from transformers import image_utils, video_utils
    except Exception as exc:
        if verbose:
            print(f"Could not import transformers compatibility modules: {exc}")
        return False

    if hasattr(image_utils, "VideoInput"):
        return True

    if not hasattr(video_utils, "VideoInput"):
        if verbose:
            print("transformers.video_utils.VideoInput was not found.")
        return False

    image_utils.VideoInput = video_utils.VideoInput
    if verbose:
        print("Added compatibility alias: transformers.image_utils.VideoInput")
    return True


def fix_import(content):
    """
    Split the import: keep most things from image_utils, move VideoInput to video_utils
    """
    # Pattern to match the from transformers.image_utils import block
    pattern = r'from transformers\.image_utils import \(((?:[^)]+|\n)+)\)'
    
    match = re.search(pattern, content)
    if not match:
        # Try single line import
        pattern = r'from transformers\.image_utils import ([^\n]+)'
        match = re.search(pattern, content)
        if not match:
            return content, False
    
    imports_block = match.group(1)
    
    # Check if VideoInput is in the imports
    if 'VideoInput' not in imports_block:
        return content, False
    
    # Split imports and remove VideoInput
    imports = [imp.strip().rstrip(',') for imp in re.split(r',\s*|\n', imports_block)]
    imports = [imp for imp in imports if imp]  # Remove empty strings
    
    image_utils_imports = [imp for imp in imports if 'VideoInput' not in imp]
    
    # Build the new import statements
    if image_utils_imports:
        new_image_import = 'from transformers.image_utils import (\n    ' + ',\n    '.join(image_utils_imports) + '\n)'
    else:
        new_image_import = ''
    
    new_video_import = 'from transformers.video_utils import VideoInput'
    
    # Replace in content
    if new_image_import:
        replacement = new_video_import + '\n' + new_image_import
    else:
        replacement = new_video_import
    
    new_content = content[:match.start()] + replacement + content[match.end():]
    
    return new_content, True


def fix_videollama3_import(hf_home=None, verbose=True):
    """
    Fix VideoInput import in VideoLLaMA3 cached model files and runtime modules.
    
    Args:
        hf_home: Path to HuggingFace cache directory. If None, uses HF_HOME env var or default.
        verbose: Whether to print progress messages.
    
    Returns:
        dict: Results with keys 'success', 'fixed_count', 'already_fixed_count', 'message'
    """
    runtime_patched = ensure_transformers_videoinput_compat(verbose=verbose)

    # Get HF cache location
    if hf_home is None:
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    
    # HuggingFace may escape hyphens as '_hyphen_' in paths, so check both patterns
    search_paths = [
        Path(hf_home) / 'modules' / 'transformers_modules' / 'DAMO-NLP-SG' / 'VideoLLaMA3-7B',
        Path(hf_home) / 'modules' / 'transformers_modules' / 'DAMO_hyphen_NLP_hyphen_SG' / 'VideoLLaMA3_hyphen_7B',
    ]
    
    # Find all existing paths and collect files from all of them
    existing_paths = [path for path in search_paths if path.exists()]
    
    if not existing_paths:
        msg = f"VideoLLaMA3-7B not found in any expected location"
        if verbose:
            print(msg)
            print("Model not downloaded yet - run this after first download attempt.")
        return {
            'success': runtime_patched,
            'fixed_count': 0,
            'already_fixed_count': 0,
            'runtime_patched': runtime_patched,
            'message': msg,
        }
    
    if verbose:
        print(f"Searching in {len(existing_paths)} location(s):")
        for path in existing_paths:
            print(f"  - {path}")
    
    # Fix modeling_videollama3_encoder.py: wrap flash_attn import in try/except
    # flash_attn may be installed but fail to import if aiter is missing (ROCm env)
    for search_path in existing_paths:
        for enc_file in search_path.rglob('modeling_videollama3_encoder.py'):
            with open(enc_file, 'r') as f:
                content = f.read()
            old_block = (
                'if is_flash_attn_2_available():\n'
                '    from flash_attn import flash_attn_varlen_func\n'
                'else:\n'
                '    flash_attn_varlen_func = None'
            )
            new_block = (
                'if is_flash_attn_2_available():\n'
                '    try:\n'
                '        from flash_attn import flash_attn_varlen_func\n'
                '    except Exception:\n'
                '        flash_attn_varlen_func = None\n'
                'else:\n'
                '    flash_attn_varlen_func = None'
            )
            if old_block in content:
                with open(enc_file, 'w') as f:
                    f.write(content.replace(old_block, new_block))
                if verbose:
                    print(f"Fixed flash_attn import in: {enc_file}")
            elif new_block in content:
                if verbose:
                    print(f"✓ flash_attn import already fixed: {enc_file}")

    # Find all image_processing_videollama3.py files from all existing paths
    files = []
    for search_path in existing_paths:
        files.extend(list(search_path.rglob('image_processing_videollama3.py')))
    
    if not files:
        msg = "No image_processing_videollama3.py files found."
        if verbose:
            print(msg)
        return {
            'success': runtime_patched,
            'fixed_count': 0,
            'already_fixed_count': 0,
            'runtime_patched': runtime_patched,
            'message': msg,
        }
    
    # Fix each file
    fixed_count = 0
    already_fixed_count = 0
    
    for file_path in files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if already fixed
        if 'from transformers.video_utils import VideoInput' in content:
            if verbose:
                print(f"✓ Already fixed: {file_path}")
            already_fixed_count += 1
            continue
        
        # Apply fix
        if verbose:
            print(f"Fixing: {file_path}")
        fixed_content, changed = fix_import(content)
        
        if not changed:
            if verbose:
                print(f"  No VideoInput import found, skipping")
            continue
        
        # Backup and write
        with open(str(file_path) + '.backup', 'w') as f:
            f.write(content)
        
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        if verbose:
            print(f"✓ Fixed! Backup: {file_path}.backup")
        fixed_count += 1
    
    msg = f"Fixed {fixed_count} file(s), {already_fixed_count} already fixed"
    return {
        'success': True,
        'fixed_count': fixed_count,
        'already_fixed_count': already_fixed_count,
        'runtime_patched': runtime_patched,
        'message': msg,
    }


if __name__ == '__main__':
    result = fix_videollama3_import(verbose=True)
    exit(0 if result['success'] or result['already_fixed_count'] > 0 else 1)
