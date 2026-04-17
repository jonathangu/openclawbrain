# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-028`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3b96b2a97bdbabf2a2491696460b8da77dc242a25c47533759e1ca69d544c781`
- fixture hash: `sha256-32449d86eb6b142eb11e1d76d43e4c37d62e87233bae5b870977e6a064fa97e1`
- score hash: `sha256-e1f89765e299dfec5086ed55faefbe6327a6343304b5ad8190ba877697cd0529`
- bundle hash: `sha256-6735f01e13fa5e9d6225c5a6ba74ba9d74f7f4605b4bc14748b8a36217022c3b`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4b14dd3575bcaf16e76897e36504d083be01ba320a2077714c9a7749ba84f112 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-320b51164e14c94bbe91eb8528c0c0f2245860f223484230069547c2efbd1fef |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cc5b01fec705e485fcde70c15ea585310aa4b2c1af7e0b1aa3732f72aec2c099 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-000b36de9a41c4af8c3681b2fdb23780613626df90dc223073ddc89b7791e644 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b7b1a42e | sha256-6241d04ac3a48953ba7fbacae7f0cc8fbdd5f159cb469027f24eb7062b8e266b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b7b1a42e | sha256-f78e2a1c3e72145b1c73b6f937507426d54aec103fbebd3ee85fb04216d48465 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-589bf593 | sha256-6895c34611b8fb2f216aec65f42b85df633be4e75a5de74ec2013db4ef30ea03 |
