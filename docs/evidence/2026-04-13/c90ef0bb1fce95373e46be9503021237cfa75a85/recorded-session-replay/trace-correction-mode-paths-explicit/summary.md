# Recorded Session Replay Proof Bundle

- trace id: `trace-correction-mode-paths-explicit`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fab75339653bcbb0cb0d29b79ed0413420391a2bac49b3daf77ba52271dbcc40`
- fixture hash: `sha256-27dd9ff4afc45f85fa83146b5f2f7b2ecf21060b7787f28311720e3beef95163`
- score hash: `sha256-a9276fe5945f71057087bde65ea5634c3c3e3de1f2d98ea398f3414974e8ffd3`
- bundle hash: `sha256-d887e4564f499ce966c25da635bef8a540c2f2653d59a4a70767f332e70d3c7f`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/3 | 0 | 0 | 2 | 1 | 0 | sha256-3bd1527e8f6d8b45bb21926050c3d0ca5b84e1544b44b11e823a7df9dc0523ab |
| vector_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-700cc37777f9f1d2ee9ab2f1f25e70ab0372b4275febfccd43af87bb60d16478 |
| graph_prior_only | 2 | 2 | 3/3 | 0 | 0 | 2 | 1 | 0 | sha256-f4de41e9b362500d715362937b15d96fb6f2259f95cbbbbde568f8e7ace479be |
| learned_route | 2 | 2 | 3/3 | 1 | 1 | 2 | 1 | 0 | sha256-b56530fa46ce2f0d3b73b1a1a183f82e908d13a54682bae34a5274d7342b735d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | mode-paths-turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | mode-paths-turn-2 | 0 | no | 0/2 | no | no | none | none |
| vector_only | mode-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-0b86adf3 | sha256-fdc6921ed1b31871d4f2776200d2f057cd0a5645e33eeb4c775b801aabfe5e07 |
| vector_only | mode-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-0b86adf3 | sha256-fdc6921ed1b31871d4f2776200d2f057cd0a5645e33eeb4c775b801aabfe5e07 |
| graph_prior_only | mode-paths-turn-1 | 100 | yes | 1/1 | no | no | pack-0b86adf3 | sha256-fdc6921ed1b31871d4f2776200d2f057cd0a5645e33eeb4c775b801aabfe5e07 |
| graph_prior_only | mode-paths-turn-2 | 100 | yes | 2/2 | no | no | pack-0b86adf3 | sha256-fdc6921ed1b31871d4f2776200d2f057cd0a5645e33eeb4c775b801aabfe5e07 |
| learned_route | mode-paths-turn-1 | 100 | yes | 1/1 | no | yes | pack-0b86adf3 | sha256-fdc6921ed1b31871d4f2776200d2f057cd0a5645e33eeb4c775b801aabfe5e07 |
| learned_route | mode-paths-turn-2 | 100 | yes | 2/2 | yes | no | pack-41435f1f | sha256-cfb70f4e92bfe49a72a314425e02e54153b1338d50496ae92169af561a2eef9a |
