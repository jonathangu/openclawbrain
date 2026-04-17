# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-15894236758cffd6885df088771bc9158a039d8e6dca7ba37e0c0ae93f2bb22c`
- fixture hash: `sha256-897b7fdc496e16305fc54601a8aba44f23b5322a6b7036c26e9f447dc3d9e950`
- score hash: `sha256-37b8bd284ae83bdc1f8c3755376446b946bfce2383d1a5185438410e1a4d31d0`
- bundle hash: `sha256-eb3e01458e4aeba288fa6d2db64698a0c148c6aedbdc0253fbbf1e28ce46dd90`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-05b9f1a0d0ad4a80c5a15a8f7ef9c5d2527f8753fe005026d39ad6af8199556b |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cecbc93623336bd6216e202554c5cff34ffc2a7e0a059b7144fb9bc4e538dea9 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dfb07a34af1508ae087aa7fe4de7318871f3ad8009db33b7cca7e7468a9966fc |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5479073f4c8c859d17e3b83defed133b64bfd0445eceea8d3d1ea66759c70793 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-48cb10f9 | sha256-31ce5f9fea5832071f9a4ed4ff723c06848f70d91e2fa45bdc8f5ae9f491e5c1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-48cb10f9 | sha256-f124132f49bd544577539892be19295d1faca3108562a3c3f857c64db9a0c7e7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0d4b9b8e | sha256-e888850309c3c8dd839fcff3b93cd83c8df310d694bec3df8224b4b97df2c9a8 |
