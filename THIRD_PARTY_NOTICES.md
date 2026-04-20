# Third-Party Notices

This repository's original source code is distributed under the MIT License.
Third-party components remain under their respective licenses and are not
relicensed under MIT by this repository.

## Distribution model

- The Python `wheel` / `sdist` for this project does not vendor `ndlocr-lite`.
  It declares `ndlocr-lite` as an external dependency that is installed
  separately.

## Included or bundled third-party components

### ndlocr-lite

- Copyright: National Diet Library, Japan
- Upstream: https://github.com/ndl-lab/ndlocr-lite
- License: Creative Commons Attribution 4.0 International (`CC BY 4.0`)
- Notes:
  - This project depends on `ndlocr-lite` at install time.

### deskew_ht (vendored)

- Local path: `src/ai_ocr_pipeline/_vendored/deskew_ht`
- Upstream basis: https://github.com/kakul/Alyn/tree/master/alyn
- Original upstream license: MIT
- Local modifications notice:
  - The vendored `LICENSE_deskew_ht` file states that modifications in the
    bundled variant are additionally marked under `CC BY 4.0` by the National
    Diet Library, Japan.

## Where to look

- Repository source distribution:
  - Root `LICENSE`
  - Root `THIRD_PARTY_NOTICES.md`
  - Vendored `src/ai_ocr_pipeline/_vendored/deskew_ht/LICENSE_deskew_ht`
