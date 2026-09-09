param(
    [int]$Port = 8501,
    [switch]$UseSystemPython
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param(
        [string]$RepoRoot,
        [bool]$PreferSystemPython
    )

    if (-not $PreferSystemPython) {
        $candidates = @(
            (Join-Path $RepoRoot "venv\Scripts\python.exe"),
            (Join-Path $RepoRoot ".venv\Scripts\python.exe")
        )
        foreach ($candidate in $candidates) {
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python executable not found. Install Python 3.11+ first."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path (Join-Path $repoRoot "app.py"))) {
    throw "app.py not found. Run this script from the project repository."
}

$pythonExe = Resolve-PythonExe -RepoRoot $repoRoot -PreferSystemPython:$UseSystemPython.IsPresent
$env:PYTHONUTF8 = "1"

function Test-PythonModule {
    param(
        [string]$PythonExe,
        [string]$ModuleName
    )
    & $PythonExe -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "[Civitas] Project root: $repoRoot"
Write-Host "[Civitas] Python: $pythonExe"
Write-Host "[Civitas] Streamlit port: $Port"
Write-Host "[Civitas] Runtime policy: online API first, per-request fallback to offline mode."
Write-Host "[Civitas] Open http://127.0.0.1:$Port after startup."

$docxOk = Test-PythonModule -PythonExe $pythonExe -ModuleName "docx"
$pdfOk = Test-PythonModule -PythonExe $pythonExe -ModuleName "reportlab"
if (-not $docxOk -or -not $pdfOk) {
    Write-Warning "[Civitas] Report export dependencies are partially missing."
    if (-not $docxOk) { Write-Host "  - missing python-docx (DOCX export disabled)" }
    if (-not $pdfOk) { Write-Host "  - missing reportlab (PDF export disabled)" }
    Write-Host "  Main simulation and defense pages will still run."
}

if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY) -and [string]::IsNullOrWhiteSpace($env:ZHIPU_API_KEY)) {
    Write-Warning "[Civitas] API keys not found. Running in full offline fallback mode."
} else {
    Write-Host "[Civitas] API keys detected. Online inference will be attempted first."
}

& $pythonExe -m streamlit run app.py --server.port $Port
