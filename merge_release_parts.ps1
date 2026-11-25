$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'release-parts.json') -Raw | ConvertFrom-Json
$outputPath = Join-Path $PSScriptRoot $manifest.archive
$temporaryPath = $outputPath + '.partial'
if ((Test-Path -LiteralPath $outputPath) -or (Test-Path -LiteralPath $temporaryPath)) { throw 'Output already exists. Use a new folder to avoid overwriting files.' }
foreach ($part in $manifest.parts) {
    $partPath = Join-Path $PSScriptRoot $part.name
    if (-not (Test-Path -LiteralPath $partPath)) { throw "Missing part: $($part.name)" }
    if ((Get-Item -LiteralPath $partPath).Length -ne $part.bytes) { throw "Size mismatch: $($part.name)" }
    Write-Host "Checking $($part.name)"
    if ((Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $part.sha256) { throw "Checksum mismatch: $($part.name)" }
}
$output = [System.IO.File]::Open($temporaryPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $manifest.parts) {
        $inputFile = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $part.name))
        try { $inputFile.CopyTo($output, 8388608) } finally { $inputFile.Dispose() }
    }
} finally { $output.Dispose() }
if ((Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.sha256) { throw 'Merged archive checksum mismatch. Partial output retained for inspection.' }
Move-Item -LiteralPath $temporaryPath -Destination $outputPath
Write-Host "PASS: restored original complete archive: $outputPath"
