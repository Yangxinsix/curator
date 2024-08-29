# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Add new license as the code is transferred from original repository.

### Changed

- Now MLCalculator will calculate stress if only virial is given.
- Now calculator will be instantiated from configuration file.

## [1.1.1] - 2024-08-29

### Added

- v1.1.1 Add per species metrics and per atoms metrics.
- v1.1.1 Add a small config file example for stress training.

### Changed

- Now using per species and per atoms metrics for default training.

### Fixed

- Fix stress training errors. Now using virials to calculate stress as default.