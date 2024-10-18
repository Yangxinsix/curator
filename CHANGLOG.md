# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Add new license as the code is transferred from original repository.
- Add feature extractor things to extract features from model and calculate mahalanobis distance with provided dataset.

### Changed

- Now MLCalculator will calculate stress if only virial is given.
- Now calculator will be instantiated from configuration file.
- Change default log directory. Optimize it for wandb.
- Sync metrics for DDP training.

### Fixed

- Fix MACE training bugs. Now training MACE model is way much easier and efficient!
- Fix curator-select not working for MACE and NequIP issue.

## [1.1.1] - 2024-08-29

### Added

- v1.1.1 Add per species metrics and per atoms metrics.
- v1.1.1 Add a small config file example for stress training.

### Changed

- Now using per species and per atoms metrics for default training.

### Fixed

- Fix stress training errors. Now using virials to calculate stress as default.
