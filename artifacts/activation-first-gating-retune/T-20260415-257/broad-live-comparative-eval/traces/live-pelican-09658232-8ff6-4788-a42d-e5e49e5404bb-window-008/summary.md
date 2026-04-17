# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-dc7aa6c27637d299d6eae706b4fc67a2a2a7b4de77c818a562317ac57ca7ac6f`
- fixture hash: `sha256-3d9a8c7638fdfa743ac7a63700e6bcceed5b6728eed1bfa78f1b2db0ab28c6de`
- score hash: `sha256-d5b017cdb07a759f9ff1ea812cd537b58fdbdddfecaf2d0a3a77aa423757b2ff`
- bundle hash: `sha256-be62eef6bd6394197db19c935f74bda87edad7b2ae91d2ab0447a25ea7fbfeb4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6474375f5bcb6a5860753785382ca496af4bf19e7ca31262302583c0776eda20 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c052903f4f7afcdba01c61393a51998855eb13f3aa37758ed9f6eccf3b550cca |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e5b67b00fd0ba114d82e61075f51cc1c97091f560202d0f4977b0a87483204cc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-d579ff0eeae255647028a484f4096bad013da17fd6c4cdfd15b1e974805ef643 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f0f97d9 | sha256-ae85065a733fdea3e85751aa5a291c3a48eb6fc7b4b776f11da7edff58b11d3f |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8f0f97d9 | sha256-acf825f1c16fead73c1b839bb1e372cb7269c3f7aa603289484ffca50ba8365d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-a98e07c6 | sha256-51e7ce76e798f0440a3d40e1ab4b9f88c299524a998de852d6f2e52f6459e197 |
