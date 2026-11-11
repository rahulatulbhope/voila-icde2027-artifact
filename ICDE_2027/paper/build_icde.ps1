[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot 'source')
try {
    & pdflatex -interaction=nonstopmode -halt-on-error icde2027.tex
    if ($LASTEXITCODE -ne 0) { throw 'First pdflatex pass failed.' }

    & bibtex icde2027
    if ($LASTEXITCODE -ne 0) { throw 'bibtex failed.' }

    & pdflatex -interaction=nonstopmode -halt-on-error icde2027.tex
    if ($LASTEXITCODE -ne 0) { throw 'Second pdflatex pass failed.' }

    & pdflatex -interaction=nonstopmode -halt-on-error icde2027.tex
    if ($LASTEXITCODE -ne 0) { throw 'Third pdflatex pass failed.' }

    Copy-Item -LiteralPath icde2027.pdf -Destination (Join-Path $PSScriptRoot 'icde2027_author_edited.pdf') -Force
    Write-Output "Built $((Join-Path $PSScriptRoot 'icde2027_author_edited.pdf'))"
}
finally {
    Pop-Location
}
