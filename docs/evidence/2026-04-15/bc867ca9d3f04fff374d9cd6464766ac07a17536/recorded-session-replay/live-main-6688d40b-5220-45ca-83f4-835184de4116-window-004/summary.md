# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-2210bd8aa54ff55f81e90c13af23591578b0c820206054d3d91e01211b88bae7`
- fixture hash: `sha256-a562aa7a1ac863aa823f236bdbc816afd7b8d62760a47e5474f699f78bdac5e9`
- score hash: `sha256-1835b8b4c13533777324d0748ef0e2d21ebc9304f9c78f60018e26215d6712c8`
- bundle hash: `sha256-ace5d8cec38e59db6108100671a72bf53966d930b8a9111f8c05bf3153dcc1f9`

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
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-978d9a347818d9cd4cb7c9ba8a535d14b879c1548288724d519b453bdd5c458b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c71012e092cfbc75e1712c8987d1269f01ee00a22e4363720aa352d2d198919d |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cacefdd57b4891d23faff098ac6644ad89a2a3b1ff69cc1064814304e09f36f6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-29576d99 | sha256-76a1cbd241c70cc75645d2f0e9ea99e3a67802405c13851807e861765ed1d7fc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-29576d99 | sha256-eccd5b34fd1fad67e2e47a2fc45456bdec6f2b15f3e77d7ae2e65093cecdf914 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-29576d99 | sha256-76a1cbd241c70cc75645d2f0e9ea99e3a67802405c13851807e861765ed1d7fc |
