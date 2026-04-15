# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e98818ee1fbfe6af19470aba80e3474e972af078ccda49d0b283bf9b3f9cdf05`
- fixture hash: `sha256-d657d23463bf41cd4159e478f5223c1f2880e97d1b0706959b1e80d3f0d4e745`
- score hash: `sha256-ac9c0d94d152316a2abb18359b63b9b4e7ac687c6e2590305c371e5ab0de7714`
- bundle hash: `sha256-3f355daa6c7cca7884e9481f862bc3f9633557d442317b2fbb16b7769120f3ba`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eaf48763997381e6d9ac148445f4fc78050abde4363c03acd4f6f65040d7cf98 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-353dd0d81bddfcce72296af5a2c7bd562bc0d9df8ef88867c9d1148f3cc2e655 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d449ae01bc9485791692ada164eff18bc736336728757a0aeaa8266c8e94a2dc |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-66bed25edbab2c4b8bc20e09f42bafe14ae8f97bb71bbc42ea6410e7ba7d87bf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d0a373b0 | sha256-d69842f37dec95dceee1d319aaeaadcf6c6f44715fd00da380569fc5a926b713 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d0a373b0 | sha256-e9ed26adcbfbaa85520ab920cf8d4d0e21149c950716248a06ff13181f0194c5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-d0a373b0 | sha256-d69842f37dec95dceee1d319aaeaadcf6c6f44715fd00da380569fc5a926b713 |
