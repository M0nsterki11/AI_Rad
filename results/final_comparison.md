# Interni test vs. vanjski test

| Model | Interni accuracy | Vanjski accuracy | Pad accuracy | Interni macro precision | Vanjski macro precision | Interni macro recall | Vanjski macro recall | Interni macro F1 | Vanjski macro F1 | Pad macro F1 | Interni sec/doc | Vanjski sec/doc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 89.33% | 28.00% | 61.33% | 89.54% | 15.71% | 89.33% | 28.00% | 89.37% | 19.33% | 70.04% | 0.0020s | 0.0402s |
| XLM-RoBERTa | 99.33% | 76.00% | 23.33% | 99.35% | 66.67% | 99.33% | 76.00% | 99.33% | 69.29% | 30.04% | 0.0122s | 0.0240s |
| LayoutLMv3 | 100.00% | 56.00% | 44.00% | 100.00% | 39.09% | 100.00% | 56.00% | 100.00% | 43.61% | 56.39% | 0.0526s | 0.0588s |
