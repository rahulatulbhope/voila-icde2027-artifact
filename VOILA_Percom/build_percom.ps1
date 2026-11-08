param(
    [ValidateSet("working", "submission", "both")]
    [string]$Mode = "both",
    [switch]$StrictSubmissionReferences
)

$ErrorActionPreference = "Stop"
$SourceDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $SourceDirectory
try {
    function Invoke-PerComBuild {
        param([bool]$Submission, [bool]$StrictReferences)

        if ($Submission) {
            $JobName = "percom2027_submission"
            $TexInput = "\def\PerComSubmission{1}\input{percom2027.tex}"
        }
        else {
            $JobName = "percom2027"
            $TexInput = "percom2027.tex"
        }

        & pdflatex -interaction=nonstopmode -halt-on-error "-jobname=$JobName" $TexInput
        if ($LASTEXITCODE -ne 0) { throw "Initial pdflatex pass failed for $JobName" }
        & bibtex $JobName
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed for $JobName" }
        & pdflatex -interaction=nonstopmode -halt-on-error "-jobname=$JobName" $TexInput
        if ($LASTEXITCODE -ne 0) { throw "Second pdflatex pass failed for $JobName" }
        & pdflatex -interaction=nonstopmode -halt-on-error "-jobname=$JobName" $TexInput
        if ($LASTEXITCODE -ne 0) { throw "Final pdflatex pass failed for $JobName" }

        $Warnings = Select-String -Path "$JobName.log" -Pattern "undefined references|undefined citations|Citation .* undefined|Reference .* undefined"
        if ($Warnings) {
            if ($StrictReferences) {
                $Warnings | ForEach-Object { Write-Error $_.Line }
                throw "Unresolved references remain in $JobName"
            }
            $Warnings | ForEach-Object { Write-Warning $_.Line }
        }
        Write-Host "Built $JobName.pdf"
    }

    if ($Mode -in @("working", "both")) {
        Invoke-PerComBuild -Submission $false -StrictReferences $true
    }
    if ($Mode -in @("submission", "both")) {
        Invoke-PerComBuild -Submission $true -StrictReferences $StrictSubmissionReferences.IsPresent
    }
}
finally {
    Pop-Location
}
