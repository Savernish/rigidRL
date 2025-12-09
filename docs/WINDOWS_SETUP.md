# Windows Development Setup Guide

You want to switch to native Windows development to fix rendering issues. Here is how to set up the project on Windows.

## 1. Prerequisites

Install the following tools:
1.  **Visual Studio Build Tools** (or VS Community): You do **NOT** need to use the Visual Studio IDE.
    *   Install "Desktop development with C++".
    *   This gives you the **MSVC Compiler** (`cl.exe`) which CMake uses.
    *   You can continue using **VS Code** (and Antigravity!) on Windows.
2.  **CMake**: [Download Installer](https://cmake.org/download/). Add it to your system PATH.
3.  **Python 3.10+**: [Download Installer](https://www.python.org/downloads/). Ensure "Add Python to PATH" is checked.
4.  **Git**: [Download Installer](https://git-scm.com/download/win).

## 2. Managing Libraries (The Easy Way)
We strongly recommend using **vcpkg** to install C++ libraries like Eigen and SDL2 on Windows.

1.  Clone vcpkg (e.g., into `C:\dev\vcpkg`):
    ```powershell
    git clone https://github.com/microsoft/vcpkg.git C:\dev\vcpkg
    .\vcpkg\bootstrap-vcpkg.bat
    ```
2.  Install Dependencies:
    ```powershell
    C:\dev\vcpkg\vcpkg.exe install eigen3 sdl2:x64-windows
    ```

## 3. Python Dependencies
Open PowerShell in the project root:
```powershell
pip install pybind11 numpy
```

## 4. Building the Project
We provide a `compile.bat` script that uses `pip` and `CMake` to build the project automatically.

1.  Open PowerShell in the project root.
2.  Run:
    ```powershell
    .\compile.bat
    ```
    Or manually:
    ```powershell
    pip install -e .
    ```

## 5. Running
The build system automatically updates the installation. You can run examples from anywhere:

```powershell
python examples/test_engine_loop.py
python examples/falling_box_visual.py
```

## 6. Development with Antigravity (VS Code)
You can absolutely continue using **VS Code** and **Antigravity** on Windows!

1.  Open this folder in VS Code on Windows.
2.  Install extensions: **C/C++** and **CMake Tools**.
3.  Select your Kit:
    *   Press `Ctrl+Shift+P` -> "CMake: Select a Kit".
    *   Choose **"Visual Studio Community 2022 Release - amd64"**.
4.  Configure:
    *   `Ctrl+Shift+P` -> "CMake: Configure".
    *   (Ensure you set the `CMAKE_TOOLCHAIN_FILE` for vcpkg in `.vscode/settings.json` or via command arguments).

This way, you keep your familiar agent environment while getting the benefits of native Windows rendering!
