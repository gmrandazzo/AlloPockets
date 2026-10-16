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


def install_pyrosetta():
    print(">>> Checking PyRosetta...")
    try:
        import pyrosetta  # noqa: F401

        print("PyRosetta is already installed.")
    except ImportError:
        print(
            "WARNING: PyRosetta is licensed by RosettaCommons for NON-COMMERCIAL ACADEMIC USE ONLY."
        )
        print("By proceeding with this installation, you agree to their academic license terms.")
        print("If you are a commercial user, please cancel this installation (Ctrl+C).")
        print("Installing pyrosetta-installer and PyRosetta (this may take a few minutes)...")
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "pyrosetta-installer"], check=True)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()",
            ],
            check=True,
        )


def install_fpocket(bin_dir):
    print(">>> Installing fpocket...")
    if shutil.which("fpocket", path=str(bin_dir)) or shutil.which("fpocket"):
        print("fpocket is already installed.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Cloning fpocket (v4.2.3)...")
        subprocess.run(
            ["git", "clone", "--branch", "4.2.3", "https://github.com/Discngine/fpocket.git"],
            cwd=tmpdir,
            check=True,
        )
        fpocket_dir = Path(tmpdir) / "fpocket"
        print("Compiling fpocket...")
        make_args = ["make"]
        if sys.platform == "darwin":
            import platform

            machine = platform.machine().lower()
            if machine == "arm64":
                make_args.append("ARCH=MACOSXARM64")
            else:
                make_args.append("ARCH=MACOSXX86_64")
        subprocess.run(make_args, cwd=fpocket_dir, check=True)
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
        print("Cloning hh-suite (v3.3.0)...")
        subprocess.run(
            ["git", "clone", "--branch", "v3.3.0", "https://github.com/soedinglab/hh-suite.git"],
            cwd=tmpdir,
            check=True,
        )
        hh_dir = Path(tmpdir) / "hh-suite"
        build_dir = hh_dir / "build"
        build_dir.mkdir()
        release_dir = hh_dir / "release"
        print("Compiling hh-suite...")
        subprocess.run(
            [
                "cmake",
                f"-DCMAKE_INSTALL_PREFIX={release_dir}",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                "..",
            ],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(["make", "-j", "4"], cwd=build_dir, check=True)
        subprocess.run(["make", "install"], cwd=build_dir, check=True)

        hhmake_path = release_dir / "bin" / "hhmake"
        if hhmake_path.exists():
            shutil.copy(hhmake_path, bin_dir / "hhmake")
            print(f"hhmake installed to {bin_dir / 'hhmake'}")
        else:
            print("Failed to find compiled hhmake binary.")


def install_freesasa(bin_dir):
    print(">>> Checking freesasa...")
    if shutil.which("freesasa", path=str(bin_dir)) or shutil.which("freesasa"):
        print("freesasa is already installed.")
        return

    print("freesasa is missing. It is a required C binary.")
    if sys.platform == "darwin":
        print("Attempting to install freesasa via Homebrew...")
        if shutil.which("brew"):
            subprocess.run(["brew", "install", "freesasa"], check=False)
        else:
            print("Homebrew is not installed. Please install freesasa manually (e.g. from source).")
    elif sys.platform.startswith("linux"):
        print("\n" + "=" * 70)
        print(" ERROR: 'freesasa' is missing!")
        print(" AlloPockets requires FreeSASA with JSON support to run predictions.")
        print(" Please install it manually via your package manager:")
        print("   Ubuntu/Debian: sudo apt-get install freesasa")
        print("   CentOS/RHEL: sudo yum install freesasa")
        print("=" * 70 + "\n")
        raise RuntimeError(
            "freesasa is not installed. Please run the package manager command above."
        )


def install_dssp(bin_dir):
    print(">>> Checking dssp (mkdssp)...")

    # Check if existing mkdssp is version >= 4.4.0
    import subprocess
    import shutil
    import sys
    import urllib.request

    mkdssp_path = shutil.which("mkdssp", path=str(bin_dir)) or shutil.which("mkdssp")
    if mkdssp_path:
        try:
            result = subprocess.run([mkdssp_path, "--version"], capture_output=True, text=True)
            if "mkdssp version" in result.stdout:
                version_str = result.stdout.split("version")[1].strip()
                v_parts = [int(x) for x in version_str.split(".")]
                if len(v_parts) >= 2 and (v_parts[0] > 4 or (v_parts[0] == 4 and v_parts[1] >= 4)):
                    print(f"Valid dssp version {version_str} is already installed.")
                    return
                else:
                    print(f"Found outdated dssp version {version_str}. Minimum required is 4.4.0.")
        except Exception:
            print("Could not verify dssp version. Will attempt to install.")

    print("dssp/mkdssp requires complex dependencies (libcifpp, boost).")
    if sys.platform == "darwin":
        print("Attempting to install dssp via Homebrew...")
        if shutil.which("brew"):
            subprocess.run(["brew", "tap", "brewsci/bio"], check=False)
            subprocess.run(["brew", "install", "dssp"], check=False)
        else:
            print("Homebrew is not installed. Please install dssp manually.")
    elif sys.platform.startswith("linux"):
        print("Downloading static mkdssp 4.4.0 binary for Linux...")
        url = "https://github.com/PDB-REDO/dssp/releases/download/v4.4.0/mkdssp-4.4.0-linux-x64"
        dest = bin_dir / "mkdssp"
        try:
            urllib.request.urlretrieve(url, dest)  # nosec B310
            dest.chmod(0o755)
            print("Successfully installed static mkdssp 4.4.0.")
        except Exception as e:
            print(f"Failed to download static dssp binary: {e}")
            raise RuntimeError("Could not install dssp automatically.")


def run_installation():
    print("Initializing AlloPockets External Dependencies Installation...")
    check_prerequisites()
    bin_dir = get_bin_dir()

    try:
        install_pyrosetta()
        install_fpocket(bin_dir)
        install_hhsuite(bin_dir)
        install_dssp(bin_dir)
        install_freesasa(bin_dir)
    except subprocess.CalledProcessError as e:
        print(f"\nError during compilation/installation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

    print(f"\nInstallation complete! Binaries are stored in {bin_dir}")
    print("AlloPockets will automatically detect and use these binaries.")
