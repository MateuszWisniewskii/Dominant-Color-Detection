Biologically inspired artificial intelligence project, a program that prints the dominant color from a photo.


How to use:

1. Create virtual environment:
    python3 -m venv venv

2. Activate it: 
    Depending on your current platform the invocation of the activation script wil be different:

    PLATFORM    SHELL       COMMAND TO ACTIVATE VIRTUAL ENVIRONMENT

    POSIX:      bash/zsh    source <venv>/bin/activate
                fish        source <venv>/bin/activate.fish
                csh/tcsh    source <venv>/bin/activate.csh
                pwsh        <venv>/bin/Activate.ps1

    Windows:    cmd.exe     <venv>\Scripts\activate.bat
                PowerShell  <venv>\Scripts\Activate.ps1
    
    <venv> must be replaced by the path to the directory containing the virtual environment

    More information about creating virtual environment you can find under that website:
    https://docs.python.org/3/library/venv.html

3. Install requirements:
    pip install -r requirements.txt

4. run program: 
    python3 main.py
