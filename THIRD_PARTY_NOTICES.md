# Third-party notices

FaunaVox is released under the MIT License (see `LICENSE`). It includes, or is built with, the following
third-party work. Their notices must be kept with any copy of FaunaVox, including binary builds of
`faunavox.dll` and `cvsynth`: `native/build.ps1` copies this file into `native/dist` for that reason.

## Compiled into the library

### CMU Pronouncing Dictionary

`native/data/cmudict.dict` is compiled into `faunavox.dll` and `cvsynth`, and read by `english_phrase.py`. It's
used to turn English text into phrases for the `speak` intent. BSD-style license (also in
`native/data/cmudict.LICENSE`):

```
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Rust crates

Statically linked into `faunavox.dll` and `cvsynth`. Each is offered under a choice of licenses; FaunaVox uses
each under the MIT license, whose text follows the list.

| Crate | Version | Authors | Licenses offered | Source |
|---|---|---|---|---|
| serde | 1.0.229 | Erick Tryzelaar, David Tolnay | MIT or Apache-2.0 | https://github.com/serde-rs/serde |
| serde_core | 1.0.229 | Erick Tryzelaar, David Tolnay | MIT or Apache-2.0 | https://github.com/serde-rs/serde |
| serde_json | 1.0.154 | The serde-rs developers | MIT or Apache-2.0 | https://github.com/serde-rs/json |
| itoa | 1.0.19 | David Tolnay | MIT or Apache-2.0 | https://github.com/dtolnay/itoa |
| zmij | 1.0.23 | David Tolnay | MIT | https://github.com/dtolnay/zmij |
| memchr | 2.8.3 | Andrew Gallant (Copyright (c) 2015 Andrew Gallant), bluss | MIT or Unlicense | https://github.com/BurntSushi/memchr |

Used only while compiling, and not part of any build: serde_derive, syn, quote and proc-macro2 (MIT or Apache-2.0),
and unicode-ident ((MIT or Apache-2.0) and Unicode-3.0).

```
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## Sources of parameters and methods

No code from these is included. They're credited because some of FaunaVox's values and methods follow them.

- **Klattara**, Copyright (c) 2026 RetroBunn, MIT License. Some consonant values, such as the voice-bar level
  and the exaggerated formant loci of stops, follow its tables and notes, which are themselves written from
  published papers.
- **The NRL letter-to-sound rules**: Elovitz, Johnson, McHugh and Shore, "Automatic translation of English text to
  phonetics by means of letter-to-sound rules", Naval Research Laboratory report 7948 (1976), a work of the US
  government. `native/data/english_rules.txt` is a transcription of them.
- **Published research** for the model's values: Dennis Klatt's formant synthesizer (formant transitions and
  loci); Lorin Wilde's thesis on fricatives (MIT, 1995) for consonant levels and spectra; Standard Southern British
  English monophthong formant means from three speakers, for the SSBE vowels; and Morton's motivation-structural
  rules for the intent templates.

## Used, not included

The Python reference engine (`create_vocalization.py`) imports numpy and scipy (BSD-3-Clause), numba (BSD-2-Clause)
and, optionally, sounddevice (MIT). They're installed separately and not distributed with FaunaVox.
