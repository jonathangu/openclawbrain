# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-2082423c993f4562d57afa1c4deadf165c9e6fa7355488f8eb4fa202c43db419`
- bundle hash: `sha256-bf5e91ae0d817f2d90610387a8e065a5476bb07f61dd291f2f88bf945fa9c376`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e54eea5dd476d45e5e7ab52a9b0ed2c646fc990677d2858d9966f3baecd8936 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-1f31e8b9f4b7c070c4bfbce078c2596551b22da4f396ad501b41e92311cf3d6a |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc75e2087487288fa6ab84c0eb5c2690ea485f38378b76567443bb5d35871fe7 |
| learned_route | 1 | 1 | 3/3 | 1 | 0 | 1 | 0 | 2 | sha256-efa2c4e59244c37afc81ae2f5718c23211084e2f130d9b8fbd80a8185b93ab55 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-9e0990bc | sha256-eed473168c79b5a2282466f6512e4f3540a8e4008b3ff1a404b0cc9238fc3c95 |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-9e0990bc | sha256-9ae90283173bec8cc2591cb979e091ebf03c256c13414f1e197fe2957c2580aa |
| learned_route | turn-1 | 100 | yes | 3/3 | yes | no | pack-f28ed983 | sha256-4538bc4e42ff76a5b382dfd79acddd7184ad01c2fae490a9e186b0fd29d8c0ed |
