# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2210bd8aa54ff55f81e90c13af23591578b0c820206054d3d91e01211b88bae7`
- fixture hash: `sha256-a562aa7a1ac863aa823f236bdbc816afd7b8d62760a47e5474f699f78bdac5e9`
- score hash: `sha256-748d1584556ed14e07226839bfbaf5724071d804b7f87ea32168865d7fe10d6b`
- bundle hash: `sha256-d7f8edcda46baa86461a78c2e6cd7a103b98847f91c9840a76c1c52fbcc2e3ed`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cbbf8eff09f23d982b9af94fdc9d383c8e6e748daa65afe086a31e073a634311 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6eed4d636b5eee2a8e299babe398dcf12c1eae65513e275d22194cedca33abe9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e3b3231b38382da13aea3ade9723d13b05eca1f489903e3af76f14f11a619ff4 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-e1894c679f25b9a122866a2c879599643686411c185529e352b1823b5d742b08 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d640e17 | sha256-7233def8e96adee2473a7b2c6099556fce435408e1644d1db6155992e82587a4 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0d640e17 | sha256-638ba2fc54343d83ef09642042123d84001d8b2bc8e03185c0c393ac8434febf |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-0a6a444c | sha256-c00db37debbdd2cf209cdab6f754983b7ea1ab8342071b1c3f71c174d64ce8e3 |
