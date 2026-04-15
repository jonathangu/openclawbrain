# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1d9740289fc2adbace7590e78dff24d1d94c6a419d6474e3af27754996da05a`
- fixture hash: `sha256-b9420b72d3a2c2c9c62adbc0b7f3ef24407bf200cf73b9c382cce44e2d33fe6a`
- score hash: `sha256-04d3345e884df48945c652db47b8511a530af179d96ae401ff5ef8ee00e20934`
- bundle hash: `sha256-751b41b974ea0952d99acd8f0445757f0683f720e1ebd19dfbe1d41f1caa0503`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a36b1da37ad1b5e7a8a6bf9a89b082e6da9affb9cefe62c4630aaf0bc52cbd76 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d4b0bdc83b643c9208769e850bc53dd9378f21d9bbbb969cd0f8711a3121606c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fbf43ce8f4c347c9a17b85d5ae22c5a1bc5e35f3253c7735efcd870290ec79a8 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-0e9b9aec328136a1133e29981927257c8042aab917d2ec46d152709ce39b7bb4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1b762f3b | sha256-a0278416b90557aeac69f0925d2d2c7e1c3bd572038fbd94881ff5824a085353 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-1b762f3b | sha256-b7f54a8c401ed06e69e277def5b1919cc824fee51515dce9f214f41cbe16aec3 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1b762f3b | sha256-a0278416b90557aeac69f0925d2d2c7e1c3bd572038fbd94881ff5824a085353 |
