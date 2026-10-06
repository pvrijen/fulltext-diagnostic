# Third-Party Notices

## BELT (bert_for_longer_texts)

The `belt_plus/` directory is derived from BELT, released under the MIT
License. It has been extended for this work with knowledge distillation,
section-aware processing, and additional chunk-pooling strategies.

- Source: https://github.com/mim-solutions/bert_for_longer_texts
- Initial authors: Michał Brzozowski, Marek Wachnicki

Original license text, reproduced as required by the MIT License:

```
MIT License

Copyright (c) 2023 MIM SOLUTIONS Sp. z o.o.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Pre-trained models

The default encoder, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
(formerly PubMedBERT), is distributed by Microsoft under the MIT License via
the Hugging Face Hub. Other encoders referenced in `experiments/` are subject
to their respective licenses on the Hub.
