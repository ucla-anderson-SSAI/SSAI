#!/usr/bin/env python3
"""Fix the router files to properly reference data paths and clean up imports."""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(BASE_DIR, 'api')

def fix_router_file(filepath: str):
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix the broken CORS line that got mangled
    content = re.sub(
        r'# CORS handled in main\.py.*?(?=\n[A-Z_]|\nclass|\ndef|\n#)',
        '',
        content,
        flags=re.DOTALL
    )

    # Fix DATA_DIR paths to be relative to project root
    content = re.sub(
        r'DATA_DIR\s*=\s*os\.path\.join\(os\.path\.dirname\([^)]+\)[^)]*\)',
        'DATA_DIR = "data"',
        content
    )
    content = re.sub(
        r'DATA_DIR\s*=\s*os\.path\.dirname\([^)]+\)',
        'DATA_DIR = "data"',
        content
    )

    # Fix any remaining path issues
    content = content.replace('os.path.join(os.path.dirname(__file__), "..", "data")', '"data"')
    content = content.replace('os.path.join(os.path.dirname(__file__), "data")', '"data"')

    # Remove duplicate FastAPI import if APIRouter is present
    if 'APIRouter' in content:
        # Keep only what we need
        content = re.sub(
            r'from fastapi import APIRouter,\s*FastAPI,',
            'from fastapi import APIRouter,',
            content
        )
        content = re.sub(
            r'from fastapi import FastAPI,\s*APIRouter,',
            'from fastapi import APIRouter,',
            content
        )

    # Remove StaticFiles and FileResponse if not needed
    if 'StaticFiles' not in content.replace('from fastapi.staticfiles import StaticFiles', ''):
        content = re.sub(r'from fastapi\.staticfiles import StaticFiles\n?', '', content)
    if 'FileResponse' not in content.replace('from fastapi.responses import FileResponse', ''):
        content = re.sub(r'from fastapi\.responses import FileResponse\n?', '', content)

    # Remove CORS middleware import if not used
    if 'CORSMiddleware' not in content.replace('from fastapi.middleware.cors import CORSMiddleware', ''):
        content = re.sub(r'from fastapi\.middleware\.cors import CORSMiddleware\n?', '', content)

    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Fixed {os.path.basename(filepath)}")

def main():
    for filename in os.listdir(API_DIR):
        if filename.endswith('.py') and filename != '__init__.py':
            fix_router_file(os.path.join(API_DIR, filename))

if __name__ == '__main__':
    main()
