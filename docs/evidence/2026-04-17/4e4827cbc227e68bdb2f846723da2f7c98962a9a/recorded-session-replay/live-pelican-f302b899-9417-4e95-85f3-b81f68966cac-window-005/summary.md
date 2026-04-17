# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6aeb45f46257078a73e31a3ca01fc811e5a3a9b2828328d1595fb41ae1cb1b87`
- fixture hash: `sha256-b90901422fe4620c22145acdd76fedd90d08a07ca2636957ff33166af8db8c6b`
- score hash: `sha256-682cff7065e247a49e8e80031cf81a050f844f4e1b87bb167e26a696a10be569`
- bundle hash: `sha256-a3514bf4ae901aa8f1b0208cdceda6c0ff1f9e49d9c2f9fa9c46fb584097516b`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f9c00df70ff9e588e665c6961063a6f0105a883c9e9bd2b1d2f815eef1057f7d |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4602b6d61ecca672073b190a0b79e0737a7c1df7a6fa4fc4d6c8d9ddc22a99a7 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-652367f1c85a02f0046de5ae98f85e9ebd2498be5f09879cc693c5d9583c5376 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-ad7d97587f63ff431d519c36dfc1125bfb9f487d74bc564f2549867121c8ee27 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f0d4c49e | sha256-b066c04e99eb7dd0f5f347c2728bf74d25f99b1ba50cf15b8e51754d9f5bb339 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f0d4c49e | sha256-eb1df76282b11176ec328f6d57a135dce1322d8d13a3d634fbbb135329e3dbd1 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-46bdefeb | sha256-ecbf307bdb596c27364ec60293e9a7036e723e399776cf338a81725e7dfe06aa |
