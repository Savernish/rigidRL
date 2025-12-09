running build_ext
-- Selecting Windows SDK version 10.0.26100.0 to target Windows 10.0.26200.
-- Configuring done (0.4s)
-- Generating done (0.3s)
-- Build files have been written to: C:/Users/enbiy/diff_sim/build/temp.win-amd64-cpython-312/Release
MSBuild version 17.14.23+b0019275e for .NET Framework

  engine.cpp
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(511,40): error C2039: 'rot': is not a member of 'Body' [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
      C:\Users\enbiy\diff_sim\diff_sim_core\include\engine\body.h(26,7):
      see declaration of 'Body'
  
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(511,35): error C2530: 'theta': references must be initialized [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,45): error C2665: 'cos': no overloaded function could convert all the argument types [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
      C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cmath(344,46):
      could be 'long double cos(long double) noexcept'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,45):
          'long double cos(long double) noexcept': cannot convert argument 1 from 'Tensor' to 'long double'
              C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,49):
              No user-defined-conversion operator available that can perform this conversion, or the operator cannot be called
      C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cmath(88,40):
      or       'float cos(float) noexcept'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,45):
          'float cos(float) noexcept': cannot convert argument 1 from 'Tensor' to 'float'
              C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,49):
              No user-defined-conversion operator available that can perform this conversion, or the operator cannot be called
      C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\corecrt_math.h(509,35):
      or       'double cos(double)'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,45):
          'double cos(double)': cannot convert argument 1 from 'Tensor' to 'double'
              C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,49):
              No user-defined-conversion operator available that can perform this conversion, or the operator cannot be called
      C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cmath(837,1):
      or       'double cos(_Ty) noexcept'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,45):
          'double cos(_Ty) noexcept': could not deduce template argument for '__formal'
              C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cstddef(36,39):
              'std::enable_if_t<false,int>' : Failed to specialize alias template
      C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,45):
      while trying to match the argument list '(Tensor)'
  
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(512,35): error C2530: 'cos_t': references must be initialized [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,45): error C2665: 'sin': no overloaded function could convert all the argument types [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
      C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cmath(522,46):
      could be 'long double sin(long double) noexcept'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,45):
          'long double sin(long double) noexcept': cannot convert argument 1 from 'Tensor' to 'long double'
              C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,49):
              No user-defined-conversion operator available that can perform this conversion, or the operator cannot be called
      C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cmath(256,40):
      or       'float sin(float) noexcept'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,45):
          'float sin(float) noexcept': cannot convert argument 1 from 'Tensor' to 'float'
              C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,49):
              No user-defined-conversion operator available that can perform this conversion, or the operator cannot be called
      C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\corecrt_math.h(517,35):
      or       'double sin(double)'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,45):
          'double sin(double)': cannot convert argument 1 from 'Tensor' to 'double'
              C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,49):
              No user-defined-conversion operator available that can perform this conversion, or the operator cannot be called
      C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cmath(838,1):
      or       'double sin(_Ty) noexcept'
          C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,45):
          'double sin(_Ty) noexcept': could not deduce template argument for '__formal'
              C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include\cstddef(36,39):
              'std::enable_if_t<false,int>' : Failed to specialize alias template
      C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,45):
      while trying to match the argument list '(Tensor)'
  
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(513,35): error C2530: 'sin_t': references must be initialized [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(559,61): error C2397: conversion from 'float' to 'int' requires a narrowing conversion [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(567,61): error C2397: conversion from 'float' to 'int' requires a narrowing conversion [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(584,57): error C2397: conversion from 'float' to 'int' requires a narrowing conversion [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
C:\Users\enbiy\diff_sim\diff_sim_core\src\engine\engine.cpp(585,57): error C2397: conversion from 'float' to 'int' requires a narrowing conversion [C:\Users\enbiy\diff_sim\build\temp.win-amd64-cpython-312\Release\forgeNN_cpp.vcxproj]
Traceback (most recent call last):
  File "C:\Users\enbiy\diff_sim\setup.py", line 57, in <module>
    setup(
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\site-packages\setuptools\__init__.py", line 117, in setup
    return distutils.core.setup(**attrs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\site-packages\setuptools\_distutils\core.py", line 186, in setup
    return run_commands(dist)
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\site-packages\setuptools\_distutils\core.py", line 202, in run_commands
    dist.run_commands()
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\site-packages\setuptools\_distutils\dist.py", line 1002, in run_commands
    self.run_command(cmd)
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\site-packages\setuptools\dist.py", line 1104, in run_command
    super().run_command(command)
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\site-packages\setuptools\_distutils\dist.py", line 1021, in run_command
    cmd_obj.run()
  File "C:\Users\enbiy\diff_sim\setup.py", line 24, in run
    self.build_extension(ext)
  File "C:\Users\enbiy\diff_sim\setup.py", line 55, in build_extension
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
  File "C:\Users\enbiy\AppData\Local\Programs\Python\Python312\Lib\subprocess.py", line 413, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['cmake', '--build', '.', '--config', 'Release', '--', '/m']' returned non-zero exit status 1.
