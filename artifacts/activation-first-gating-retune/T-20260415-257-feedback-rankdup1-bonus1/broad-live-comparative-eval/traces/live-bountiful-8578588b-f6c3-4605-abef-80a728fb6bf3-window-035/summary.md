# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-035`
- winner mode: `graph_prior_only`
- trace hash: `sha256-36169ed7c6b7573ef644e5e670a60e6f7c1c993fe52a7f13686f76bf635cc43f`
- fixture hash: `sha256-16d6e6092a3d5f8800f19820b3a256e739ead32b4381f2896d9aeccf372e3bde`
- score hash: `sha256-c8a60b2f774fa41500bd88b123bdcd52a10d70ec6e9c3c44835a6118a695cc1a`
- bundle hash: `sha256-20f0725337e4ca43a17fc910be6e70e14f7a096707dcdb35bfa71381ce55a8d7`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7c9a32ff2de979b024c8bfc14bcb8fa72199e8333c117ab27753dd3626b13edb |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-606579963d2aa3d11fcabbdb0cb79031802b6a70a7b51ed4c4b4304d7aad92b1 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-2545a67ea75867e7576d223ff1a1045071ba7c496a34c116b5b419b6ff3c06fe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-3892913c15880e925f708d284b470a90e0b31f5efa4a54dbf99eaaa72fea4db1 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-e2266ae1 | sha256-3dcb000374dab5cbb1dfc548aae6f1efb0faf830734af73aa6abaef1e3a11764 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-e2266ae1 | sha256-9001f3d200dfa1dda7e1f7388a8ccd247be73351aff7e30e7923eb5815404485 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e2266ae1 | sha256-3dcb000374dab5cbb1dfc548aae6f1efb0faf830734af73aa6abaef1e3a11764 |
