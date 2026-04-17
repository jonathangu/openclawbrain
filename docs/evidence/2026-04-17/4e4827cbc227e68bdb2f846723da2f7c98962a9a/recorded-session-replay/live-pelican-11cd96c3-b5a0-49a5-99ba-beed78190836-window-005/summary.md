# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e98818ee1fbfe6af19470aba80e3474e972af078ccda49d0b283bf9b3f9cdf05`
- fixture hash: `sha256-d657d23463bf41cd4159e478f5223c1f2880e97d1b0706959b1e80d3f0d4e745`
- score hash: `sha256-7b49060c45fda8c87783eea148a4a418ce6f2d4a8cd7cbd4953a75cdbedeb320`
- bundle hash: `sha256-e0da4e682a338dcf277a27c69422b30a392537d052a7163447d52b0d81153391`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eaf48763997381e6d9ac148445f4fc78050abde4363c03acd4f6f65040d7cf98 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8dfc627b7c27ea83c55194c19186c21867fc288ba605e1d70aef0166eb8d311c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b515490a8a23b2ef02ab4babc0126c529c7f743d4053a8ffe44e579c38ca13fe |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-40442e6dc9cfab107ce5d625104a2debeb2b7627b631e1e0796181ffc1ddd13c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b8754948 | sha256-86fc92a14d8d0d7bb6a63d0e5eb2fc755e1f36ab07014157cbf84b6d6210f723 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b8754948 | sha256-49b3f2cd19631e48ac6af9a4c7cc27239dce9ccaa6140dc834ce9b96fe88e6e7 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-83e9dec5 | sha256-dbfa4e3bc094a47f28252456f8f124f77cff3ef18bd5d5920e250a5528ad15a7 |
