# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-015`
- winner mode: `graph_prior_only`
- trace hash: `sha256-04fe85c8a179be229d8c68dec97a25113ffce0ef409792233e0ccc1c65106721`
- fixture hash: `sha256-66a77cd573b5398a7b3b4867686fe20ef718501f851c3ff410c457c68968fa97`
- score hash: `sha256-83a3132634fc0c1cc037111adaa693850b2a3a94bea01071df7e97425e94cbbd`
- bundle hash: `sha256-107dadad5291e10f9dde92464dd47d703e6c1ea25cf18a60a96a1702d7eb110d`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ae930e12c21c7056f67f547427d9cdedef7d7970b442aa81b3fdb75182425c80 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-97e7045c11834f39505b5a4df86c56247ff266eefc0298e12d3f0a25f3a293d1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-634b15d4db95d6bb36638b603d52a7371e38615b1b335e173165342784e28576 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-a61a45e9a72a84b67b6d934d6cb6f2b437fa4ddc1e70ddec3841815fe57e5700 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b37c0ef | sha256-acfe19f3004a9ae7282ce17fa1f406d0210e2b25cf91b13258391497de50699f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b37c0ef | sha256-df3f4c5cb26cca328a2851ecb1860e0acb050d341bd554637b889b0bcaa77a6b |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b8b4f07e | sha256-8f4a7a733cfdbae89c0da5da37eee0217da9b30315bf4bdfc02ecb770101c64f |
