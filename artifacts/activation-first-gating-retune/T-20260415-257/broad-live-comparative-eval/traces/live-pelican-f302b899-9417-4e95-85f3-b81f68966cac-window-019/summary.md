# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-a95260c17a69374ef7a9ff20490cb415b09868b4babdf035ba541b6d82beb5bf`
- fixture hash: `sha256-ff74599acd0d3d5ad2046fb7795a787fe8fa0e70837c98ae65f89838fc9f50e9`
- score hash: `sha256-805bf64104f17c3362abee4bde76adc1859e988b9f157b7a315e90de52f5343b`
- bundle hash: `sha256-2ce29cdb872a6124c421d895ed52cec704163efcdf0710994f63062287bce4f1`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30f22d5abba84d169a9ef0f72b28eb7bd4c2afa26a7910c928c371f416decf04 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c15add7134f8f3786f9cd5016deeba6e999a955db75572eedfee5ca8480945b4 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-492a0e5b4abb0fadf2abc1118b55ea1e89fcb4b72c3aac2b7da0e0f9af21c208 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-f3e8ca67f56a895df28406122c607648a7bb6f63b290c88a57e5d3c1838595ac |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-14146e7f | sha256-334e7616d0666a1fc5a2c23e9fcfde1e588892fd2803348ddbaa4aa2b59cffd1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-14146e7f | sha256-7075e541558448672dd13b926dfdb2f6e62cd51d51592fb81bc5f7f184d28028 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-10e97c38 | sha256-99996125570bce029963ab190aa80f8f9e50f3bc37d540dcdd2553d8fc084644 |
