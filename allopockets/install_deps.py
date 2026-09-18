import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

def get_bin_dir():
    home = Path.home()
    bin_dir = home / ".allopockets" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir

def check_prerequisites():
    missing = []
    for tool in ["git", "make", "cmake", "gcc"]:
        if not shutil.which(tool):
            if tool == "gcc" and sys.platform == "darwin" and shutil.which("clang"):
                continue
            missing.append(tool)
    if missing:
        print(f"Error: Missing system prerequisites: {', '.join(missing)}")
        if sys.platform == "darwin":
            print("Please install them using: xcode-select --install && brew install cmake")
        elif sys.platform.startswith("linux"):
            print("Please install them using: sudo apt-get install build-essential cmake git")
        sys.exit(1)

def install_fpocket(bin_dir):
    print(">>> Installing fpocket...")
    if shutil.which("fpocket", path=str(bin_dir)) or shutil.which("fpocket"):
        print("fpocket is already installed.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Cloning fpocket...")
        subprocess.run(["git", "clone", "https://github.com/Discngine/fpocket.git"], cwd=tmpdir, check=True)
        fpocket_dir = Path(tmpdir) / "fpocket"
        print("Compiling fpocket...")
        subprocess.run(["make"], cwd=fpocket_dir, check=True)
        bin_path = fpocket_dir / "bin" / "fpocket"
        if bin_path.exists():
            shutil.copy(bin_path, bin_dir / "fpocket")
            print(f"fpocket installed to {bin_dir / 'fpocket'}")
        else:
            print("Failed to find compiled fpocket binary.")

def install_hhsuite(bin_dir):
    print(">>> Installing hh-suite (hhmake)...")
    if shutil.which("hhmake", path=str(bin_dir)) or shutil.which("hhmake"):
        print("hhmake is already installed.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Cloning hh-suite...")
        subprocess.run(["git", "clone", "https://github.com/soedinglab/hh-suite.git"], cwd=tmpdir, check=True)
        hh_dir = Path(tmpdir) / "hh-suite"
        build_dir = hh_dir / "build"
        build_dir.mkdir()
        release_dir = hh_dir / "release"
        print("Compiling hh-suite...")
        subprocess.run(["cmake", f"-DCMAKE_INSTALL_PREFIX={release_dir}", ".."], cwd=build_dir, check=True)
        subprocess.run(["make", "-j", "4"], cwd=build_dir, check=True)
        subprocess.run(["make", "install"], cwd=build_dir, check=True)
        
        hhmake_path = release_dir / "bin" / "hhmake"
        if hhmake_path.exists():
            shutil.copy(hhmake_path, bin_dir / "hhmake")
            print(f"hhmake installed to {bin_dir / 'hhmake'}")
        else:
            print("Failed to find compiled hhmake binary.")

def install_dssp(bin_dir):
    print(">>> Checking dssp (mkdssp)...")
    if shutil.which("mkdssp", path=str(bin_dir)) or shutil.which("mkdssp") or shutil.which("dssp"):
        print("dssp is already installed.")
        return

    print("dssp/mkdssp requires complex dependencies (libcifpp, boost).")
    if sys.platform == "darwin":
        print("Attempting to install dssp via Homebrew...")
        if shutil.which("brew"):
            subprocess.run(["brew", "install", "dssp"], check=False)
        else:
            print("Homebrew is not installed. Please install dssp manually.")
    elif sys.platform.startswith("linux"):
        print("On Linux, please install dssp via your package manager:")
        print("  Ubuntu/Debian: sudo apt-get install dssp")
        print("  CentOS/RHEL: sudo yum install dssp")

def run_installation():
    print("Initializing AlloPockets External Dependencies Installation...")
    check_prerequisites()
    bin_dir = get_bin_dir()
    
    try:
        install_fpocket(bin_dir)
        install_hhsuite(bin_dir)
        install_dssp(bin_dir)
    except subprocess.CalledProcessError as e:
        print(f"\nError during compilation/installation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
        
    print(f"\nInstallation complete! Binaries are stored in {bin_dir}")
    print("AlloPockets will automatically detect and use these binaries.")

