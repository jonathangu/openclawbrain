# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-032`
- winner mode: `graph_prior_only`
- trace hash: `sha256-92a53a83b75391e6ea2e19694e75cc46987c1fd7f2482c72c3850eb3ee758d5b`
- fixture hash: `sha256-a7a70c06edd57e7fef42061ce44261270b10f99213ced50cea189f13c03e8e7a`
- score hash: `sha256-d21764d6891789bdbc1b9c05ccd497327146f175ac36c42992740dbd9700bd5b`
- bundle hash: `sha256-dd95437ed5e94815335318a88d35b12ae25593c9ec1edd9d96f586f69aed422c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-467d90a6c748c6c78cf3c7ceb933156139020979bf5f7ad7e3a8103479da429a |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-750e1ceb08818b4a070a5b798c04f994324e35b54ffded89778ff028b1a5a947 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a5e1d4c73821025f55765c25f5a80564fc70612ae5384bc9de8d60e29f90469 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-de0958935a6db4f63a24af6258966a38d039ecf873b3daf43fc1a35d91cd3d91 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e6f36bff | sha256-81c6e775e506545dd15e0096447f40a231e72d19f8bb4bfbd66434bb9ab4512e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-e6f36bff | sha256-4492582f1f2f89202bf27e5eeabb763e086b706c350e70be28abef6393fe6d60 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-04578014 | sha256-e5ea659acf2bde89133cd1385b47f0c354607a15f652ec65a627e28906751052 |
