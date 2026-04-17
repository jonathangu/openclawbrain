# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-34bdf8f02ec779363fa4c8a951850c78f5f147002a61b16879fb9ea405e3f18a`
- fixture hash: `sha256-fcd33f2f91aff8f0b5df7411e8b655364e067521c4da8a9896b3ab460088a1d2`
- score hash: `sha256-6a3a3ac88debefcb0a9687379b234bab3177696390baf3c8b5e9cc2033c7d1be`
- bundle hash: `sha256-d230091f2cc523efe514b19bbf524ca9bab5c6c0297cf752d0f4ff3078f7258a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-59aa7e4d7b28a1f5c691ae41334f8392171bd8edbe01742338c29f7ed9b2609b |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4c379dc6d80c0c68d4d9e4711656a03e64596fcc21f79890c0ecf44ea3e272e9 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6d503fc20541e7f22a3ed699a3734b795d0ca145f4ea2070d9770fecd0a0028f |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-ebe7e546c8976ec6c777a14e015165cdc313e87aa90177c89fdcfa124931ec67 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5b54615b | sha256-e359c10985e2993cca92753e99019c5647794f124e9d94c7b497d7f10ccff132 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-5b54615b | sha256-9649131dcdf2e97e12b1903d0f77ba986aff3a1747afaabad27557ed1dd93547 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-060f04a4 | sha256-62f44c73cfe02c340676881a14c39af61c0cb6cdd438b98d96c785c79d74e2d9 |
