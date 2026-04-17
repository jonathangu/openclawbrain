# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2210bd8aa54ff55f81e90c13af23591578b0c820206054d3d91e01211b88bae7`
- fixture hash: `sha256-a562aa7a1ac863aa823f236bdbc816afd7b8d62760a47e5474f699f78bdac5e9`
- score hash: `sha256-a786dc09f88cfdee85da3e7505807e1167e0ae6bdb723285d1b3a79ec0836279`
- bundle hash: `sha256-99ef2514b52527995194f520c38072d281eed40efd6503eb0e9ee86980913f19`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cbbf8eff09f23d982b9af94fdc9d383c8e6e748daa65afe086a31e073a634311 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8249f0ce1a76dc38679078cafd6e241863e5f24691143aabacb08b6ba18e0fa2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-95877134b78bd71419f6af4924aea3ad65504da39589aa84b62da3516d45366c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-787966def1a324559f9aba2ba3aa6c0bdc27b281008ded4b38413eabea5bbe16 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-efa214f8 | sha256-963bb74dba6a728fb2a1caaf56dbd21a147e363a4dfbfec40872028f218db6ed |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-efa214f8 | sha256-9323e58253e2e6d50233c2203f8c03ac50cc11ab00ad767d407bc9caf1994f83 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-eca84b2d | sha256-a7c1a9c2cf956c67f37b1b35e72a5ca65aa11eeae72f2683b33b83a5be6ab142 |
