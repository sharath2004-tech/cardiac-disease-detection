import json

# Load the notebook
with open(r'd:\hackathons\heartsense-ai-main\cardiac-disease-detection\1_patient_wise_cross_validation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the buggy SimCLR_Model class
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the buggy get_features cell
        source_str = ''.join(source)
        if 'self.encoder.relu' in source_str and 'get_features' in source_str:
            print(f"Found buggy cell with ID: {cell.get('id', 'unknown')}")
            # Replace the buggy method
            new_source = []
            in_get_features = False
            skip_until_else = False
            
            for i, line in enumerate(source):
                # Start of get_features method
                if 'def get_features(self, x):' in line:
                    in_get_features = True
                    new_source.append(line)
                    continue
                
                # Inside get_features
                if in_get_features:
                    # Check for SimpleCNN block
                    if 'if isinstance(self.encoder, SimpleCNN):' in line:
                        new_source.append(line)
                        # Add corrected implementation
                        new_source.append('            # Conv block 1\n')
                        new_source.append('            x = self.encoder.conv1(x)\n')
                        new_source.append('            x = self.encoder.bn1(x)\n')
                        new_source.append('            x = F.relu(x)\n')
                        new_source.append('            x = self.encoder.pool1(x)\n')
                        new_source.append('            x = self.encoder.dropout(x)\n')
                        new_source.append('            \n')
                        new_source.append('            # Conv block 2\n')
                        new_source.append('            x = self.encoder.conv2(x)\n')
                        new_source.append('            x = self.encoder.bn2(x)\n')
                        new_source.append('            x = F.relu(x)\n')
                        new_source.append('            x = self.encoder.pool2(x)\n')
                        new_source.append('            x = self.encoder.dropout(x)\n')
                        new_source.append('            \n')
                        new_source.append('            # Conv block 3\n')
                        new_source.append('            x = self.encoder.conv3(x)\n')
                        new_source.append('            x = self.encoder.bn3(x)\n')
                        new_source.append('            x = F.relu(x)\n')
                        new_source.append('            \n')
                        new_source.append('            # Global pooling\n')
                        new_source.append('            x = self.encoder.global_pool(x).squeeze(-1)\n')
                        new_source.append('            \n')
                        skip_until_else = True  # Skip old buggy lines until elif
                        continue
                    
                    # Skip the old buggy implementation
                    if skip_until_else:
                        if 'elif isinstance(self.encoder, TCNClassifier):' in line:
                            skip_until_else = False
                            new_source.append('            return x\n')
                            new_source.append(line)
                        continue
                    
                    # End of get_features method (next method or class)
                    if line.strip().startswith('def ') or line.strip().startswith('class '):
                        in_get_features = False
                
                new_source.append(line)
            
            cell['source'] = new_source
            print(f"Fixed cell with {len(new_source)} lines")
            break

# Save the fixed notebook
with open(r'd:\hackathons\heartsense-ai-main\cardiac-disease-detection\1_patient_wise_cross_validation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook fixed successfully!")
