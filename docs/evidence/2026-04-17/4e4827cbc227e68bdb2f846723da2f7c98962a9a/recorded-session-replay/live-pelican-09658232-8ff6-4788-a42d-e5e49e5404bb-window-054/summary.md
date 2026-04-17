# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-054`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4f8bd6e98ba409d4b92ff33d315c90158dc9f7928f49ee95918b29862594fc07`
- fixture hash: `sha256-f2d0f492e33718dcda5e95309dd8b8ae83d2a012ce623b86565c773255e59638`
- score hash: `sha256-41da976785fb13859129b41dcdb735401d81246774e5d487e6d97658657239a1`
- bundle hash: `sha256-d58374a876484f985537d022a6aff3a0e789cadde6851a34d894569d19479e7f`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-3a49035e9fd3e0717342039595aabed753c46d3f982a6fbdc847832f0114d10f |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4f7cca1d4acc1e5744226dc665fd7a313596a6a86266b2bf105a108459d9d3ca |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-d602889fcddad43a9c6b0171536d998adc55c6a4adb889e11a1661cf72b34b1f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-1a1969411f9bab0b8501413aee5c97e5a7483690358a13923bdd467535f1744e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7e8834e1 | sha256-911dcfe9d657cf784e0f8902cdd6bd2ceaf8c10814dd28fb0d77538111ea418c |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-7e8834e1 | sha256-5af1e333dd7aa9ca21ec16bd15e45033230ab6624e3620cc2b1a7452bd9286cf |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-29e3b072 | sha256-331b9dbc4787314d45ffa27f52b3cbb84f8c0bb1731f76d485b0580e6114ba87 |
