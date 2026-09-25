# Horizontal method roster

| Method | Family | Formal role | Applicability | Source checkout | Environment |
|---|---|---|---|---|---|
| PySR | Julia/Python evolutionary SR | ancestor-matched baseline | tabular SR and legacy formulas | `MilesCranmer/PySR` | existing locked `env_1_pysr` |
| Operon | C++ GP with PyOperon binding | high-throughput GP | tabular SR and legacy formulas | `heal-research/operon` | `parallel_operon` |
| DSR | policy-gradient neural SR | neural symbolic baseline | compatible tabular SR | `dso-org/deep-symbolic-optimization-pytorch` | `parallel_dsr` |
| AI-Feynman 2.0 | physics-prior decomposition/search | physics-prior baseline | scientific known-law tasks | `SJ001/AI-Feynman` | `parallel_ai_feynman` |
| gplearn | Python DEAP GP | transparent low-barrier baseline | tabular SR and legacy formulas | `trevorstephens/gplearn` | `parallel_gplearn` |
| TF4SR | Transformer encoder-decoder | selected Transformer baseline | SRSD-compatible scientific tasks | `omron-sinicx/transformer4sr` | `parallel_tf4sr` |
| MySR | MySR/MySRCore | final target | all declared applicable strata | local MySR repositories | `env_1_mysr` |

TF4SR was selected as the Transformer representative because its official source
ships pretrained weights and a direct SRSD evaluation script. This is a protocol
choice for reproducibility and task alignment, not a claim that it is universally
the best Transformer SR system. TPSR, NeSymReS, and SymFormer remain documented
candidate extensions and are not silently mixed into the v1 main roster.
