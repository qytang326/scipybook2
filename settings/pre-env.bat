@echo off
set WINPYDIR="D:\Program Files\Python"
set WINPYVER=3.6.2
::set HOME=%WINPYDIR%\..\settings
set WINPYARCH="WIN-AMD64"
::if  "%WINPYDIR:~-5%"=="amd64" set WINPYARCH="WIN-AMD64"

rem handle R if included
::if not exist "%WINPYDIR%\..\tools\R\bin" goto r_bad
::set R_HOME=%WINPYDIR%\..\tools\R
::if %WINPYARCH%=="WIN32"     set R_HOMEbin=%R_HOME%\bin\i386
::if not %WINPYARCH%=="WIN32" set R_HOMEbin=%R_HOME%\bin\x64
:r_bad

rem handle Julia if included
::if not exist "%WINPYDIR%\..\tools\Julia\bin" goto julia_bad
::set JULIA_HOME=%WINPYDIR%\..\tools\Julia\bin\
::set JULIA_EXE=julia.exe
::set JULIA=%JULIA_HOME%%JULIA_EXE%
:::julia_bad

::set PATH=%WINPYDIR%\Lib\site-packages\PyQt5;%WINPYDIR%\;%WINPYDIR%\DLLs;%WINPYDIR%\Scripts;%WINPYDIR%\..\tools;%WINPYDIR%\..\tools\mingw32\bin;%WINPYDIR%\..\tools\mingw32\bin;%WINPYDIR%\..\tools\R\bin\i386;%WINPYDIR%\..\tools\Julia\bin;%PATH%;
set PATH=%WINPYDIR%\Lib\site-packages\PyQt5;%WINPYDIR%\;%WINPYDIR%\DLLs;%WINPYDIR%\Scripts;%PATH%;