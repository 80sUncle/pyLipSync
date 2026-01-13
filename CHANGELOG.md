# Changelog

## [0.2.1] - 2026-01-13

### Breaking Changes
- **Silence phoneme now included**: `segment.phonemes` now includes the silence phoneme. Previously, only speech phonemes were returned, with silence indicated by all targets being zero.

### Bug Fixes
- **Full audio coverage**: Fixed windowing logic to analyze audio all the way to the end. Previously, the final ~50ms of audio was not processed.

### Improvements
- Updated test suite to include silence phoneme validation
- Minor code cleanup

---

## [0.2.0] - 2026-01-12

### Breaking Changes
- **Folder structure**: `data/` renamed to `phonemes/` and `phonemes.json` renamed to `template.json`
- **Template format**: Now includes normalization stats. Old templates must be removed to generate the new one.

### New Features
- **Direct file loading**: Pass audio file paths to `extract_phoneme_segments()` without pre-loading
- **`return_seconds`**: Get segment times in seconds instead of sample indices
- **`return_audio`**: Include raw audio chunks in segment results

### Improvements
- Internal refactoring for cleaner, more maintainable code
- Enhanced test coverage with new test cases

---

## [0.1.3] - 2025-10-13

### Added
- Initial stable release
- MFCC-based phoneme detection
- Support for custom phoneme templates
- Three comparison methods: L1 Norm, L2 Norm, Cosine Similarity
