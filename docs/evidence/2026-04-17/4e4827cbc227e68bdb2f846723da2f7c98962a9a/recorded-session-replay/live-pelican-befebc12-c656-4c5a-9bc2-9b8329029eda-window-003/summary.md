# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2b0cf5e7b3f41bdf7d892185413608401c24e9c3ad252c16335ba4fe2f91cdd3`
- fixture hash: `sha256-ed8248c9b476e9fb2d02b9891cc8e11da35a8ba49c308ca9793fd2e0cd5daeaa`
- score hash: `sha256-3f56734072cf221f6b2e06fe507caf410dbe539e3f725a26f1a95ddd99163137`
- bundle hash: `sha256-e3dc1b5671acdb5732a988bbf0546219abe44d789735281ad0a196ae39ac43f1`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-368ba7ad0e0062707beb6bc226c2cae8531ed592ec4225d05a99c6ab4df81531 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-e6abc46688e39eb74caf28f20f747e5bf75e1afe1291a91f0a4a920e86e1ea85 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f91cb2e0b3e34902a396cb48ceb99ee54f7c26b2ccfa49e6f8ab78f282e6e595 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-65e60fbd3f42795e079ee1b663c7d7097a48ceb50b89d6a77de9a346bde78638 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5523864d | sha256-7626f67844467ea46e6c08afb815b73265c53dc9c4b08627d738990f2cb6106a |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-5523864d | sha256-f1542496583c71fbe6e776e5140132429e024764d5403b18a887ad2ef7027386 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-d2defb72 | sha256-cee1c29ae0eb807c509fa1d01328103758d4e15d3c1cf110bd57221552706fc0 |
