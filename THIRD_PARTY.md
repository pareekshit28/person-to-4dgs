# Source and model notices

First-party application code is AGPL-3.0; see LICENSE. Vendored source retains
its own notices and licenses. The application license does not replace them.

| Component | Source / license |
| --- | --- |
| 4DAnyone | `vendor/4danyone`, Apache-2.0; includes upstream notices for its inference extracts |
| GVHMR | `vendor/4danyone/third_party/GVHMR/LICENSE`; educational, research and non-profit use, with commercial use requiring permission |
| FreeTimeGsVanilla and FTGS browser player | `vendor/freetimegs`, AGPL-3.0 |
| CUDA rasterization dependencies | Installed at the immutable revisions listed in `vendor/SOURCES.json`; retain their upstream licenses |
| SageAttention | Optional installation at commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a` (v2.2.0); upstream Apache-2.0 |

Model weights are downloaded from the upstream pinned repositories. They are
not included in Git. SMPL-X requires a separately licensed model file obtained
by the user from https://smpl-x.is.tue.mpg.de/. The UI accepts that file without
collecting account credentials or accepting its license on the user's behalf.

The app produces reconstructed performances; the source repositories and model
licenses determine permitted use of the complete stack. This distribution
does not claim a blanket commercial license for those third-party components.

`vendor/SOURCES.json` records provenance and local changes. Large demo media was
omitted; runtime source, model download code, and license notices are included.
