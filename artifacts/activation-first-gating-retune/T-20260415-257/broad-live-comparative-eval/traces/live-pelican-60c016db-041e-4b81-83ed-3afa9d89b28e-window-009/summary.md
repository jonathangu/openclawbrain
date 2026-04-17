# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6c49ca2fe441629d93695e938073dd41facc650cc9fd301e1fa807efab482f72`
- fixture hash: `sha256-a51ac08634e3c4803c3d3973ce1a7c858ffb1429844452de0ed6e3279b36730b`
- score hash: `sha256-b7ed540fe910e1b78b20d3d0a8c8855f0652ba9a8df1f579aed219d226833668`
- bundle hash: `sha256-977d862f14f842c48068b83a1cc4e09d1d28c816ae623a61c770409ab54c1655`

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
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c2e3cc9fd6ee1604d2c526310b1ddca47a11a50e7f9573ba696d4001f01dac |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-d4e138559f161400513b0294b6d46ee5120dbc1186ae5587b8ae2d8fc278573e |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-9e2c2ecc21da4abca79e9113d3a73d865947e4bb82c48fb4211796e0deafe160 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-30f1e8bc2ff30127a8570d1db789545037d893b603b4e7f2a07a2a766cd7ba8b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1d0552bf | sha256-7a5f58c428aaf0d1d7e5d5a3ae66c082ce7b876f477cadea9b75d91b3bcb5a62 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1d0552bf | sha256-3f7b3ad4330f83e7845adf93dea9a302abc81f5270ec8a9adc12ea249a44190a |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-a07528d8 | sha256-62fe128c6698877292fed95a6a7f0ff78d2b41026a7edbf1c19398340472636e |
