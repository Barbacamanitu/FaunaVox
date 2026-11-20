# Build faunavox.dll + cvsynth.exe (release) and collect them in ./dist.
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    cargo build --release
    New-Item -ItemType Directory -Force dist | Out-Null
    Copy-Item target/release/faunavox.dll, target/release/cvsynth.exe, src/faunavox.h, unity/CreatureVoiceNative.cs, ../LICENSE, ../THIRD_PARTY_NOTICES.md dist/
    Write-Host "Built into $PSScriptRoot\dist"
} finally { Pop-Location }
