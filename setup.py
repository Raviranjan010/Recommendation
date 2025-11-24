"""
Setup script for the Content-Based Recommendation System.
This script helps users set up the environment and run the application.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_tests():
    """Run unit tests"""
    try:
        result = subprocess.run([sys.executable, "-m", "unittest", "test_recommender.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed!")
            print(result.stdout)
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_app():
    """Start the Streamlit application"""
    try:
        subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"❌ Error starting app: {e}")

def main():
    print("🎬 Content-Based Recommendation System Setup")
    print("=" * 50)
    
    print("\n🔧 Step 1: Installing requirements...")
    if not install_requirements():
        print("⚠️  Failed to install requirements. Please check your internet connection and try again.")
        return
    
    print("\n🔍 Step 2: Running tests...")
    if not run_tests():
        print("⚠️  Some tests failed. You may want to check the issues before proceeding.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n🚀 Step 3: Starting the application...")
    print("The app will open in your browser. If not, navigate to http://localhost:8501")
    start_app()

if __name__ == "__main__":
    main()