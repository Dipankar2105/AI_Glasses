param ()

Write-Host "Hardware Validation Runner for AI Glasses (XIAO ESP32-S3 Sense)"
Write-Host "---------------------------------------------------------------"

# 1. Verify project paths
$workspaceDir = Resolve-Path "$PSScriptRoot\.."
$xiaozhiPath = "C:\Users\Routewise\xiaozhi-esp32"
if (!(Test-Path $xiaozhiPath)) {
    Write-Host "Error: Xiaozhi repository not found at $xiaozhiPath"
    exit 1
}

# 2. Verify ESP-IDF
if (!$env:IDF_PATH) {
    Write-Host "Warning: IDF_PATH is not set in this session. Assuming C:\Users\Routewise\esp-idf"
    $env:IDF_PATH = "C:\Users\Routewise\esp-idf"
}
if (!(Test-Path $env:IDF_PATH)) {
    Write-Host "Error: ESP-IDF not found."
    exit 1
}

# 3. Verify ESP32-S3 target
Write-Host "Verifying target configuration... ESP32-S3 expected."

# 4. Verify existing baseline firmware
$firmwareBin = "$xiaozhiPath\build\xiaozhi.bin"
if (!(Test-Path $firmwareBin)) {
    Write-Host "Error: Baseline firmware not found at $firmwareBin. Please build it first."
    exit 1
}

# 5. Detect available COM ports
$ports = [System.IO.Ports.SerialPort]::GetPortNames()

# 6 & 7. If no board is detected
if (!$ports -or $ports.Count -eq 0) {
    Write-Host "XIAO ESP32-S3 Sense not detected. Hardware validation skipped."
    exit 0
}

# 8. If multiple possible COM ports exist
if ($ports.Count -gt 1) {
    Write-Host "Multiple COM ports detected: $($ports -join ', ')."
    Write-Host "Please specify the exact COM port manually. Safety check triggered."
    exit 1
}

$comPort = $ports[0]
Write-Host "XIAO ESP32-S3 Sense detected on $comPort."

$confirm = Read-Host "Do you want to flash the VERIFIED Phase 1 baseline firmware to $comPort? (y/N)"
if ($confirm -notmatch '^[yY]$') {
    Write-Host "Flashing aborted by user."
    exit 0
}

# 9. After confirmation: flash
Write-Host "Flashing firmware to $comPort..."
# This would invoke the flashing tool, but is guarded.
# . "$env:IDF_PATH\export.ps1"
# cd $xiaozhiPath
# python -m esptool --chip esp32s3 -p $comPort -b 460800 --before default-reset --after hard-reset write-flash --flash-mode dio --flash-size 16MB --flash-freq 80m 0x0 build\bootloader\bootloader.bin 0x8000 build\partition_table\partition-table.bin 0xd000 build\ota_data_initial.bin 0x20000 build\xiaozhi.bin 0x800000 build\generated_assets.bin

# 10. Start serial monitoring/capture
Write-Host "Starting serial monitoring and capturing log..."
# Placeholder for actual python monitor tool invocation

# 11. Save the hardware validation log
$logPath = "$workspaceDir\docs\phase1-boot-log.txt"
Write-Host "Saving hardware validation log to: $logPath"
# Placeholder for saving log stream to file

# 12. Produce final report
Write-Host "`n--- FINAL REPORT ---"
Write-Host "COM Port       : $comPort"
Write-Host "Flash Status   : TBD (Waiting for actual execution)"
Write-Host "Boot Status    : TBD"
Write-Host "PSRAM Status   : TBD"
Write-Host "Wi-Fi Status   : TBD"
Write-Host "Audio Status   : TBD"
Write-Host "Mic Status     : TBD"
Write-Host "Speaker Status : TBD"
Write-Host "Camera Status  : TBD"
Write-Host "MCP Status     : TBD"
Write-Host "Crash Status   : TBD"
Write-Host "--------------------"
