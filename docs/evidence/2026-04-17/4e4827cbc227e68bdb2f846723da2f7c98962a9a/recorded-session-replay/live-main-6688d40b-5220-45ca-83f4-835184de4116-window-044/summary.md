# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-69ce0c4e11baa36853be20e1ca688e734c8855423d37366857eb233deb6e9df0`
- fixture hash: `sha256-c3a333635db8e86be19e8bf48de8cbd13aa6939830c506cedd85267cb0e9f51f`
- score hash: `sha256-5f4586e650e1e4d1cf86fbe8130e9f1d97f804ea67d4b93f2536210177b13687`
- bundle hash: `sha256-2300783b9b3a6beed47fea1548fe6ccd06bd27f838c6627e12fbb9f9b4dbbfc8`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-653724b1c50980255f17a34150c96cf9693658619075d0cdd8b7b4b447cb2cb6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1dbe5cb2c0ff73f8bb62f58f5f47796a8442dddfef1cfd56d595b08b9011d03c |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-31a5e42e91675b0dc38831697825b0e5d7eab8b1f3c5e438fb8446e00d9ade8f |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-20aac599b0882613859f76c0900a6de9bacf7685b885a5c0170ad906997e8f42 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53184054 | sha256-09565eeb2f8c8aa38166ef29c7106e2a30c95343f66fb1e25690e4e27f9296eb |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-53184054 | sha256-8e383edc545035c5dd0b808bb4e1271490795e1216dea472cb7b2e96835f7a98 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-035e7787 | sha256-845bd0278e969ca06ac6277cd8198876073930851baa8fe9477f0cc0c493bad1 |
