# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-179`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eebf5a156737b8a1b13583833520fd34225ae0f30b4afb05ce10671f54ea2108`
- fixture hash: `sha256-63d1e3dd69e143127b58a78b17f85b8f588fdddd25950ce30e59877032c4d44a`
- score hash: `sha256-3094e35246fcf4a7624a574b6f8979eae461979fbafcdcc1c5bed075d9508366`
- bundle hash: `sha256-e152170167b549995c2ee531e82ba44f7c2e7e5b332181a5185017d5e8eaa587`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-798f7833fec1b062f8a2789c97b9a979ee4a90e5e78bf32289929e17459a82fb |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-59d6789ac9248c31561ab1e690d0ec910309d3d6d87f9900ba42eae5878cad98 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7463c4a46c5c2e12a76bab926d874937c643288974802a79a76b383327452415 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-0edc0f06d17124679448565146eff13bd057b6c42c2721a4bbb59341a6263e74 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b3abf9dd | sha256-4d5f3806c190e753c99b42a0b67c51804da9bb7641c28e4a70aab594685e81f3 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b3abf9dd | sha256-c2e97d549c27e03106ec6ec47abac7b77649818903334f293d8350eb1055d575 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-b3abf9dd | sha256-4d5f3806c190e753c99b42a0b67c51804da9bb7641c28e4a70aab594685e81f3 |
