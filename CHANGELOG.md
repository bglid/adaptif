# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## 0.1.0 -- 2026-11-15

Initial release of Python package & Rust crate.

### Added

* LMS, NLMS, & RLS implementations.
* `Algorithm` trait and `FilterBase` type for implementing custom filters.
* `src/python`: Python FFI via Py03.
* `integration/`: Integration tests for Python & Rust APIs.
* `examples/`: Examples of how to use Rust crate and Python package.
* `MAKEFILE`for codifying development.
* `.github/workflows/`: CI/CD pipelines for Python package & Rust crate.
