# Recorded Session Replay Proof Bundle

- trace id: `live-main-f554f872-80dc-4165-9326-c85c48df2834-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a9a0d976d691035c87611c3bf8262c8f24aa8cb2b2147e70d168e8ca66af5301`
- fixture hash: `sha256-5890b682144351849269c08cb811bbb472ba0970bc84bb6b237fb1117c406a77`
- score hash: `sha256-21ddf7e91a659e333fbbd595845d98cffcf7282b65b6e7b129ab2149b67faa24`
- bundle hash: `sha256-ed358ed9c1e915687266518e9ca33a41f4bfe4ff632fe0516e3db30c03566b30`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1b4a156671a0a476fe0b5ef357aaac56b6741f1cdc3373d320b9eeef3a821f69 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7111b95c76a56560dbef15e1bb6322d202639b7c36259dc660820e9aca934aa6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f35bf4e1311e4682e82cf41d4b351b871a760b095f9453de7411d3ef098e8010 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ab8ae0bd839b1091e5ac44eda18ba0bf1dea27d206cae4391b1b5c7d76219649 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-856b6272 | sha256-9c426eb5c25a6c2a51f9a2b2deaece320d750f031ad34b9424e59595c7c6128d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-856b6272 | sha256-1022b3a90b26da3d391d27ec8cfaba38bdb0fd19b557cbd8afc9bdceecc17a17 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-856b6272 | sha256-9c426eb5c25a6c2a51f9a2b2deaece320d750f031ad34b9424e59595c7c6128d |
