python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install wheel
pip install pip-tools

pip install -e .[dev,test]

$script = $PSScriptRoot+"\pip-sync.ps1"
& $script
