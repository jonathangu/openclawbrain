# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-039`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1ec62e8076ee4d1e710644be210d5ded13133f83ba7cc0a283a8ff2ec6e4b13a`
- fixture hash: `sha256-208011a3d49bd10b0f228ef3f15f5d25a591b8469fe6d29ce8deec0246fbbb48`
- score hash: `sha256-31d6aadd04348fa18b3ab740bb5885eb6cf7f26b3b71dc44d736476285d3269b`
- bundle hash: `sha256-3dac52750a7734a25be300b5cf3f58e32de81482b4d4336d79e4c4546d0e76c0`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30241ba1cbd874d0509ab1e29b9c021ef1eb69d9f017747456f3594de63d356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-8227a22d94b93a0295c3c7f4a5c8a97200c78d53a43586c1575f7a6faa6020f1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-dca819922cc218e27dd32f50689440d19fe97735e0756b0a5c82475d3704f588 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-52cc926094bf50554013aa0d270022e83f37b3d7dbab4175cfa9a11f8e9f8e8f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d090de89 | sha256-f32751180f2734031ce9bdd620713fc5ddf53b37b56d7823c86349949bcc8669 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d090de89 | sha256-ea8dbe80e0635bae03d47438f12c0f1a797c65a79079d78021a513bebea8b1db |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-92b22d9c | sha256-21e9abd77cfa41f13b7361be5b171fd72466ee61f4b351a61414bd7ee8db6f8f |
