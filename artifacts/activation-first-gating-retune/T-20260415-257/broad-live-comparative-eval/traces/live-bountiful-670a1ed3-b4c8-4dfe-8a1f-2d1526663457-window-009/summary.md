# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c58ed04d44aeb04071688c4a26c4c689e25ea007697f349c3e4c8fcbe3bda533`
- fixture hash: `sha256-ad70501e856aff4a57d924d7225c4dc64463e70da2f3e42777305ef85fb46a26`
- score hash: `sha256-317b299860c5951e66e193da6e36d9a0fdcc4e65a3669a7588a78ee10dff73d1`
- bundle hash: `sha256-9c958a404e4ec6c08ed10afb6e58ac03c881deafceb379a44990c715a2719df1`

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
- phrase hits: 0/4
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d770aca06ab90e2e0a0ead714079ce642ffbbb18580e6acfdf4fde922a74f5a7 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-39c02130a0a4b4991b9f23165c5d4554478a88ab4f2189e3ac859805f1215acb |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d6e89b13533b3092f6967a52efa7433736197b6284984c60075f5386309a2e83 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-27fb91d42b8b8de47b08695f8c6107cfaffafa7df8dbeae05d434ce0e1ee6e0f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f1d81bc7 | sha256-ab0e1834864e558517e80cc09527b867b35ed90f0874389dcfca5a5ba66371fb |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f1d81bc7 | sha256-b4838f777b4191745a3cae15540b52cda3b3a547a494a4db0516fd3557de6627 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-82e21fa4 | sha256-8ae1051e0c10401c38a6d78643231d0bcf4fed08c506423d6c114d49e91dc5aa |
