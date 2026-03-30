# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-30

### Fixed
- Fixed `handle_object_pose_command` missing `response` parameter causing ROS2 service crash (F821)
- Fixed bare `except:` clauses to use `except Exception:` in MuJoCo and Isaac Lab envs (E722)
- Fixed boolean comparison style: `== False` to `not`, `== True` to direct usage, `!= None` to `is not None` (E711/E712)
- Removed duplicate `import sapien` in ManiSkill env (F811)
- Removed unused variable assignments across codebase (F841)

### Changed
- Migrated project source code to `src/` layout, updated all internal imports to `src.` prefix
- Consolidated `requirements.txt` and `requirements-dev.txt` into a single `requirements.txt` aligned with `scripts/install.sh`
- Bumped version to 1.1.0 across all version files (pyproject.toml, src/version.py, astribot_msgs package manifests, CMakeLists)
- Removed material/texture references from "without_texture" MJCF model files
- Fixed MJCF include chain so without-texture main file references without-texture model

### Removed
- Removed auto-generated intermediate documentation from `docs/` directory
- Removed `requirements-dev.txt` (consolidated into `requirements.txt`)

## [0.1.0] - 2026-03-30

### Added
- Initial release with MuJoCo, Genesis, ManiSkill, Isaac Lab support
- ROS1 and ROS2 integration
- Multi-robot control interface
- Factory pattern for simulator backends
