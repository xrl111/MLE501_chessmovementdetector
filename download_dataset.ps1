# Download YOLO dataset
Write-Host "Downloading YOLO dataset..."
try {
    Invoke-WebRequest -Uri "https://universe.roboflow.com/ds/lEIeDLYdtb?key=ytHQpJZNeT" -OutFile "roboflow_yolo.zip"
    Write-Host "Download completed. Extracting..."
    Expand-Archive -Path "roboflow_yolo.zip" -DestinationPath "." -Force
    Remove-Item "roboflow_yolo.zip"
    Write-Host "YOLO dataset extracted successfully."
}
catch {
    Write-Host "Error downloading YOLO dataset: $_"
}

# Download Piece Detection dataset
Write-Host "Downloading Piece Detection dataset..."
try {
    Set-Location "model_dataset\PieceDetection"
    Invoke-WebRequest -Uri "https://universe.roboflow.com/ds/Km1UZuhpph?key=fcbZDX0xrI" -OutFile "roboflow_pieces.zip"
    Write-Host "Download completed. Extracting..."
    Expand-Archive -Path "roboflow_pieces.zip" -DestinationPath "." -Force
    Remove-Item "roboflow_pieces.zip"
    Write-Host "Piece Detection dataset extracted successfully."
    Set-Location "..\\.."
}
catch {
    Write-Host "Error downloading Piece Detection dataset: $_"
    Set-Location "..\\.."
}

Write-Host "Dataset download completed!"