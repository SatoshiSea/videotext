# Mac os:

:: Instala Python utilizando Homebrew

```bash
brew install python
```

:: Instala FFmpeg utilizando Homebrew, una herramienta necesaria para el procesamiento de audio y video

```bash
brew install ffmpeg
```

:: Crea un entorno virtual en la ruta especificada

```bash
python3 -m venv path/to/venv
```

:: Activa el entorno virtual (en macOS/Linux)

```bash
source path/to/venv/bin/activate
```

:: Instala la biblioteca, que se utiliza para manipular audio y demas

```bash
pip3 install -r requirements.txt
```

:: Ejecuta el script con

```bash
python main.py
```

# Windows:
:: Instala Chocolatey

```bash
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

:: Instala Python utilizando Chocolatey

```bash
choco install python
```

:: Instala FFmpeg utilizando Chocolatey, una herramienta necesaria para el procesamiento de audio y video

```bash
choco install ffmpeg
```

:: Crea un entorno virtual en la ruta especificada

```bash
python -m venv path\to\venv
```

:: Activa el entorno virtual (en Windows)

```bash
path\to\venv\Scripts\activate
```

:: Instala la biblioteca, que se utiliza para manipular audio y demas

```bash
pip install -r requirements.txt
```

:: Ejecuta el script con

```bash
python main.py
```
