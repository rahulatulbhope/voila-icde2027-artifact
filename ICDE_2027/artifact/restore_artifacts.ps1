[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DriveBundleRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$bundleRoot = (Resolve-Path -LiteralPath $DriveBundleRoot).Path

$directories = @(
    @{ Path = 'files'; Required = $true },
    @{ Path = 'raw_data'; Required = $true },
    @{ Path = 'model_artifacts'; Required = $true },
    @{ Path = 'scripts\model_artifacts'; Required = $false },
    @{ Path = 'results\percom2027'; Required = $true },
    @{ Path = 'results\percom2027_final'; Required = $true },
    @{ Path = 'results\percom2027_final_bytes'; Required = $true },
    @{ Path = 'results\percom2027_submission'; Required = $true },
    @{ Path = 'results\cache'; Required = $false }
)

foreach ($entry in $directories) {
    $relativePath = $entry.Path
    $source = Join-Path $bundleRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        if ($entry.Required) {
            throw "Missing required Drive directory: $relativePath"
        }
        Write-Warning "Missing optional Drive directory: $relativePath"
        continue
    }

    $destination = Join-Path $repoRoot $relativePath
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $destination -Recurse -Force
    Write-Output "Restored $relativePath"
}

Write-Output "Artifact restore completed under $repoRoot"
