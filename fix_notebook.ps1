# Fix the SimCLR get_features method in the notebook

$notebookPath = "d:\hackathons\heartsense-ai-main\cardiac-disease-detection\1_patient_wise_cross_validation.ipynb"

# Read the file
$content = Get-Content $notebookPath -Raw -Encoding UTF8

# Define the buggy code pattern
$buggyCode = @'
            x = self.encoder.conv1(x)\n",
    "            x = self.encoder.relu(x)\n",
    "            x = self.encoder.pool(x)\n",
    "            x = self.encoder.conv2(x)\n",
    "            x = self.encoder.relu(x)\n",
    "            x = self.encoder.pool(x)\n",
    "            x = self.encoder.conv3(x)\n",
    "            x = self.encoder.relu(x)\n",
    "            x = self.encoder.gap(x)\n",
    "            x = x.view(x.size(0), -1)\n",
    "            return x\n
'@

# Define the fixed code
$fixedCode = @'
            # Conv block 1\n",
    "            x = self.encoder.conv1(x)\n",
    "            x = self.encoder.bn1(x)\n",
    "            x = F.relu(x)\n",
    "            x = self.encoder.pool1(x)\n",
    "            x = self.encoder.dropout(x)\n",
    "            \n",
    "            # Conv block 2\n",
    "            x = self.encoder.conv2(x)\n",
    "            x = self.encoder.bn2(x)\n",
    "            x = F.relu(x)\n",
    "            x = self.encoder.pool2(x)\n",
    "            x = self.encoder.dropout(x)\n",
    "            \n",
    "            # Conv block 3\n",
    "            x = self.encoder.conv3(x)\n",
    "            x = self.encoder.bn3(x)\n",
    "            x = F.relu(x)\n",
    "            \n",
    "            # Global pooling\n",
    "            x = self.encoder.global_pool(x).squeeze(-1)\n",
    "            \n",
    "            return x\n
'@

# Replace the buggy code with fixed code
$newContent = $content -replace [regex]::Escape($buggyCode), $fixedCode

# Check if replacement was made
if ($content -eq $newContent) {
    Write-Host "ERROR: Pattern not found. Let me try a different approach..." -ForegroundColor Red
    
    # Try simpler pattern matching
    $newContent = $content -replace 'x = self\.encoder\.relu\(x\)', 'x = F.relu(x)'
    $newContent = $newContent -replace 'x = self\.encoder\.pool\(x\)', 'x = self.encoder.pool1(x)'
    $newContent = $newContent -replace 'x = self\.encoder\.gap\(x\)', 'x = self.encoder.global_pool(x)'
    
    Write-Host "Applied basic fixes..." -ForegroundColor Yellow
}

# Backup the original file
Copy-Item $notebookPath "$notebookPath.backup" -Force
Write-Host "Created backup: $notebookPath.backup" -ForegroundColor Green

# Write the fixed content
$newContent | Set-Content $notebookPath -Encoding UTF8 -NoNewline
Write-Host "Fixed notebook saved!" -ForegroundColor Green
Write-Host "Original backed up to: $notebookPath.backup" -ForegroundColor Cyan
