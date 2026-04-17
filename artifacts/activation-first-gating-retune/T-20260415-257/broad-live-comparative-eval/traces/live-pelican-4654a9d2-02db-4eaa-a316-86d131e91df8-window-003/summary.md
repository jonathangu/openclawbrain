# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-003`
- winner mode: `learned_route`
- trace hash: `sha256-871a8d0a2f1d4e43acf8de9d8e6956ae4d1ca9dd0a419c5265c96970bba52722`
- fixture hash: `sha256-219199343b7c6d3ad1312b7304ed4e0c3741109cf5c94240ae657c56e05e2f48`
- score hash: `sha256-968ce72326dc3372fc324e9fd4fdc19c74c7c7b44a2aa388cbf0872e7996481b`
- bundle hash: `sha256-7176d9aeef83a25e8b72be1bbc7b25a4e5a8f20e01b8e8e88760fc52cf0f6267`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 70 |
| 2 | vector_only | 70 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/8
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b917e276e14b121cc269645e14a5fdafe3dcdf3d48a758ee09ff6c7e3bf5cdd4 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-60bbc99ba0763b63f304d829025d001ee83d77d20096100313074fc0eec767d3 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-215708afc00a55c4081ec11651ea0ba7c5124037cc843ffc103ca9d50ea9c6a1 |
| learned_route | 1 | 1 | 1/2 | 1 | 0 | 1 | 0 | 2 | sha256-c21015bbd30ce871e15ff5a8b1302c0e7c8c2074ee600abffe82324186465ef6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-b55b0bd6 | sha256-a452830557e37625a550748ceb6c98ef12e1a758c560ffc7cba679bee04a1908 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-b55b0bd6 | sha256-a448fd39bae306187eab2a413fe228803d028ea0013f8b26f3480bb7fc0cc927 |
| learned_route | turn-1 | 70 | yes | 1/2 | yes | no | pack-aa28054b | sha256-338e88430a5df6087ea418c9358cd779d850a474627246356781023eaeec4df5 |
