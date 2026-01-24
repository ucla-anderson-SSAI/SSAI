#!/usr/bin/env python3
"""
Setup script to convert existing week/assignment apps into unified Railway deployment.
Run this once to organize all files.
"""

import os
import re
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

def convert_main_to_router(source_path: str, module_name: str) -> str:
    """Convert a main.py FastAPI app to an APIRouter module."""
    with open(source_path, 'r') as f:
        content = f.read()

    # Replace FastAPI app with APIRouter
    content = re.sub(
        r'app\s*=\s*FastAPI\([^)]*\)',
        'router = APIRouter()',
        content
    )

    # Add APIRouter import
    if 'from fastapi import' in content:
        content = content.replace(
            'from fastapi import',
            'from fastapi import APIRouter, '
        )
    else:
        content = 'from fastapi import APIRouter\n' + content

    # Replace @app. decorators with @router.
    content = re.sub(r'@app\.', '@router.', content)

    # Remove CORS middleware (handled in main.py)
    content = re.sub(
        r'app\.add_middleware\(\s*CORSMiddleware[^)]+\)[^)]*\)',
        '# CORS handled in main.py',
        content,
        flags=re.DOTALL
    )

    # Remove uvicorn.run at the bottom
    content = re.sub(
        r'if\s+__name__\s*==\s*["\']__main__["\']\s*:.*',
        '',
        content,
        flags=re.DOTALL
    )

    # Remove StaticFiles mount (handled in main.py)
    content = re.sub(
        r'app\.mount\([^)]+StaticFiles[^)]+\)',
        '# Static files handled in main.py',
        content
    )

    # Update data file paths to use /data/ directory
    content = content.replace('data/HMData.csv', 'data/HMData.csv')
    content = content.replace('data/listings.csv', 'data/listings.csv')
    content = content.replace('netflix_ratings.csv', 'data/netflix_ratings.csv')

    return content

def setup_api_modules():
    """Create API router modules from existing backends."""
    api_dir = os.path.join(BASE_DIR, 'api')
    os.makedirs(api_dir, exist_ok=True)

    # Create __init__.py
    with open(os.path.join(api_dir, '__init__.py'), 'w') as f:
        f.write('# API Routers\n')

    # Week apps
    for i in range(1, 9):
        week_dir = os.path.join(PARENT_DIR, f'week{i}-app', 'backend')
        main_py = os.path.join(week_dir, 'main.py')

        if os.path.exists(main_py):
            router_content = convert_main_to_router(main_py, f'week{i}')

            output_path = os.path.join(api_dir, f'week{i}.py')
            with open(output_path, 'w') as f:
                f.write(router_content)
            print(f"Created api/week{i}.py")

    # Assignment apps
    for i in range(1, 9):
        assignment_dir = os.path.join(PARENT_DIR, f'assignment{i}-app')
        main_py = os.path.join(assignment_dir, 'main.py')

        if os.path.exists(main_py):
            router_content = convert_main_to_router(main_py, f'assignment{i}')

            output_path = os.path.join(api_dir, f'assignment{i}.py')
            with open(output_path, 'w') as f:
                f.write(router_content)
            print(f"Created api/assignment{i}.py")

def setup_static_files():
    """Copy frontend HTML files to static directory."""
    static_dir = os.path.join(BASE_DIR, 'static')
    os.makedirs(static_dir, exist_ok=True)

    # Week frontends
    for i in range(1, 9):
        week_frontend = os.path.join(PARENT_DIR, f'week{i}-app', 'frontend', 'index.html')
        if os.path.exists(week_frontend):
            dest_dir = os.path.join(static_dir, f'week{i}')
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(week_frontend, os.path.join(dest_dir, 'index.html'))
            print(f"Copied week{i} frontend")

    # Assignment frontends
    for i in range(1, 9):
        assignment_frontend = os.path.join(PARENT_DIR, f'assignment{i}-app', 'index.html')
        if os.path.exists(assignment_frontend):
            dest_dir = os.path.join(static_dir, f'assignment{i}')
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(assignment_frontend, os.path.join(dest_dir, 'index.html'))
            print(f"Copied assignment{i} frontend")

def setup_data_files():
    """Copy data files to data directory."""
    data_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # H&M data
    hm_data = os.path.join(PARENT_DIR, 'week1-app', 'data', 'HMData.csv')
    if os.path.exists(hm_data):
        shutil.copy(hm_data, os.path.join(data_dir, 'HMData.csv'))
        print("Copied HMData.csv")

    # Netflix ratings
    netflix_data = os.path.join(PARENT_DIR, 'assignment3-app', 'netflix_ratings.csv')
    if os.path.exists(netflix_data):
        shutil.copy(netflix_data, os.path.join(data_dir, 'netflix_ratings.csv'))
        print("Copied netflix_ratings.csv")

    # Airbnb listings
    listings_data = os.path.join(PARENT_DIR, 'assignment1-app', 'data', 'listings.csv')
    if os.path.exists(listings_data):
        shutil.copy(listings_data, os.path.join(data_dir, 'listings.csv'))
        print("Copied listings.csv")

def update_frontend_api_paths():
    """Update API_BASE in frontend files to use new paths."""
    static_dir = os.path.join(BASE_DIR, 'static')

    for root, dirs, files in os.walk(static_dir):
        for file in files:
            if file.endswith('.html'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()

                # Determine which module this is
                rel_path = os.path.relpath(root, static_dir)
                if rel_path.startswith('week'):
                    api_prefix = f'/api/{rel_path}'
                elif rel_path.startswith('assignment'):
                    api_prefix = f'/api/{rel_path}'
                else:
                    continue

                # Update API_BASE
                content = re.sub(
                    r'const\s+API_BASE\s*=\s*["\'][^"\']*["\']',
                    f'const API_BASE = "{api_prefix}"',
                    content
                )

                # Also handle variations
                content = re.sub(
                    r'API_BASE\s*:\s*["\']http://localhost:\d+["\']',
                    f'API_BASE: "{api_prefix}"',
                    content
                )

                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"Updated API paths in {rel_path}/index.html")

if __name__ == '__main__':
    print("Setting up Railway deployment...")
    print("\n1. Creating API router modules...")
    setup_api_modules()

    print("\n2. Copying frontend files...")
    setup_static_files()

    print("\n3. Copying data files...")
    setup_data_files()

    print("\n4. Updating frontend API paths...")
    update_frontend_api_paths()

    print("\nSetup complete! Your project is ready for Railway deployment.")
    print("\nNext steps:")
    print("1. cd railway-deploy")
    print("2. git init && git add . && git commit -m 'Initial commit'")
    print("3. Push to GitHub")
    print("4. Connect to Railway and deploy")
