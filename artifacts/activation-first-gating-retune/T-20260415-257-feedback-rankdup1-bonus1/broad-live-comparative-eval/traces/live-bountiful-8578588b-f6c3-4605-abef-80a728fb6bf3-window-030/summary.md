# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-030`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f56abe6ae0bc3db0256fb8a29def5410559dc260bebf6c25f482767c44ad563d`
- fixture hash: `sha256-ce8c7634db2cf81b1d292959075126cdfc5e0bd2ab016e812a771d5225b9272a`
- score hash: `sha256-2bdcc3db7dccb3e581b1ce610a538f5c6bd85f48aaae4fc6cdf6b2b493c23096`
- bundle hash: `sha256-214a4be766f54104e1791582062363762ccb6909c13b5e5142ec04489a58752b`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0d6d335f3ffcc4b2fdd753f294bfd27ffc13dfaf4edadcd54e21edf30efde50 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-6ed4b42b0d560051b83b0ee9d79da8a031bd486ceed67fcf44fb1ca7c5284659 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-22459332c159c7733b283231d0f5a13caf2575d53e251740871744f4c5962cf0 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-cead1665746bf875403e1cce8b7b2c640baa4b8d6edacb21ee5d02aa8407321b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-6de024f0 | sha256-5513d195cfadfc23f1853fc89d3f07fbde6fcd524196514f4dc7caea7a785410 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-6de024f0 | sha256-dc1e1e4e9f5a3f736d3b6c45bb4c92c0e46d853357444f605147fff8b9f5fe88 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-6de024f0 | sha256-77cd24c7784711827bf4b4d494892597020c06aa9f43b8c90437cdd85b26b94b |
