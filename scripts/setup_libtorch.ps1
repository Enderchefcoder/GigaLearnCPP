# Downloads libtorch and extracts it to GigaLearnCPP/libtorch (where the build expects it).
#
# Usage (from the repository root, in PowerShell):
#   ./scripts/setup_libtorch.ps1              # CPU-only build
#   ./scripts/setup_libtorch.ps1 cu124        # CUDA 12.4 build
#   ./scripts/setup_libtorch.ps1 cu121 2.5.1  # Specific CUDA flavor + version

param(
    [string]$Flavor = "cpu",
    [string]$Version = "2.5.1"
)

$ErrorActionPreference = "Stop"

$destDir = Join-Path $PSScriptRoot "..\GigaLearnCPP"

if (Test-Path (Join-Path $destDir "libtorch")) {
    Write-Host "libtorch already exists at $destDir\libtorch"
    Write-Host "Delete that folder first if you want to reinstall."
    exit 0
}

# Windows release builds (Debug builds are a separate download; use Release for training)
$url = "https://download.pytorch.org/libtorch/$Flavor/libtorch-win-shared-with-deps-$Version%2B$Flavor.zip"

Write-Host "Downloading libtorch $Version ($Flavor) ..."
Write-Host "  $url"

$tmpZip = Join-Path $env:TEMP "libtorch_download.zip"
try {
    curl.exe -fL -o $tmpZip $url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed. Check that the version/flavor combination exists at https://pytorch.org/get-started/locally/"
    }

    Write-Host "Extracting to $destDir\libtorch ..."
    Expand-Archive -Path $tmpZip -DestinationPath $destDir

    Write-Host "Done. Configure the project with:"
    Write-Host "  cmake -S . -B build"
} finally {
    if (Test-Path $tmpZip) {
        Remove-Item $tmpZip
    }
}
