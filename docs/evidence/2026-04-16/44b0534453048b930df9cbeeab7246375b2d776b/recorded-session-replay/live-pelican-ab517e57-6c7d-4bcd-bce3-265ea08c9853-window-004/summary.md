# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e0ad9ec13f7d5b82b36a685375b9d3d24391406d595ab3f8c2b0e0a5247f79c9`
- fixture hash: `sha256-0d1840771e0444519c0d4b5e3c3b57cee2fa58fe3cd78cd2a661af1ba4273a98`
- score hash: `sha256-43df9b1947cef6ac1da98195816453cc9f7c3fbf6b89a62f9a3f04200583ec6c`
- bundle hash: `sha256-9319275e188f9c37606876fd62ef07b28c697feafd0850a2fd35fc6d419c184d`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-24e5432aabba4b367a9ab9972174d2db006f79b43849cb63eacaea39404c4061 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b7f689cf6b2f87b312f2cb878558768462f6aef0f44b5ba81b9cfce464635216 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-ab3f559e203f1d1158342d11ce51d912bba6bf708f4782640ad3d43dc9abe8e7 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-253eb956a5c5faf0b4b8f1847b40c12edbd529b0585698367e8d17801376ef96 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-da843c32 | sha256-9296831fba4ce64bc6d1282e5fe941b4cb79a453302a5dd6d26fe52cf4c9b9bd |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-da843c32 | sha256-f755117bd2b3ec9aa4a5078576a02fb002096cee165284b4ce16c34a6c09f0a1 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-a3a46e31 | sha256-12755d6056098b033ef4c548ebec358ff4f79c01aa3d6f7576b9771ec9cfb877 |
