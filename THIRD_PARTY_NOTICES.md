# Third-party notices

The MIT license covers original EvoMemory code only. Third-party projects,
models, datasets, and downloaded source retain their original licenses and terms.

- **ALMA** — <https://github.com/zksha/alma>, Apache License 2.0. EvoMemory
  loads its pinned Executor and environment adapters; setup downloads it into
  ignored `external/alma/`, preserving its license. No ALMA Meta Agent is used.
- **ALFWorld** — <https://github.com/alfworld/alfworld>. The text-game and metadata
  releases are acquired separately. This repository does not redistribute them.
- **EvoMemBench** — <https://github.com/DSAIL-Memory/EvoMemBench>. The pinned public
  CrossEp-Emb mapping is downloaded and content-verified separately.
- **MATH** — <https://github.com/hendrycks/math> and the pinned
  <https://huggingface.co/datasets/EleutherAI/hendrycks_math> distribution.
- **APPS** — <https://github.com/hendrycks/apps> and the pinned
  <https://huggingface.co/datasets/codeparrot/apps> distribution. The custom
  bounded judge documents its reference semantics; the author's reference source
  and original data are downloaded separately and are not relicensed as MIT.
- **TextWorld**, **Transformers**, **PyTorch**, **Sentence Transformers**, and
  supported model checkpoints retain their upstream licenses.

Dependencies are installed separately through package management. Review upstream
terms before redistributing datasets, model weights, or derived artifacts.
