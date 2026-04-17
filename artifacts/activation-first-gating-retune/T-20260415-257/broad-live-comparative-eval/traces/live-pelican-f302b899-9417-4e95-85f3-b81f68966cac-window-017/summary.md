# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-017`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8b83288abc1a5c66a218574e9a089abcfea75ee1de4f5813fd07c339a4e34fa2`
- fixture hash: `sha256-d84bdb541f6a2d5c8236abca3a843aa21a0e1c20f003d0fc5eb1d79b307b698e`
- score hash: `sha256-34e071b9144d4c66b99b871ef2d6029e2022ddfe3d7941470fab6dbeafeb619c`
- bundle hash: `sha256-c738c76e6d5a6eded1b85ce1f5f3483ebdfc82b44de6e5458770cb5ba886150a`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7ad0dcf523c4d76bf7e5aa9a9c949e660e04aa89d0cc57603f9d8d3b2165caa4 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-0fc90c2457684b9266c67cd570af3f3cbb0bcda7b8e28b2bc137d981fb398c34 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1a9fe528fa7790b3f3f8181dfd49905e51efc6859391d1c760f410d281d44a4c |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-bc49bb1a00099db54d9f261eea4a979fa53bff874cbd071567c539931ec22572 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ea7ee9b | sha256-46681c6f6839b373f1c42b3a004f85eabe02a6c676816b962422c7e9c53f93c2 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-2ea7ee9b | sha256-2cbae32d4b54141845442fc19248d6b7b42078ee45b5d1bc9d9337926accaff5 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-d4001e0a | sha256-ebec48bdaa1639698d328f0e223535da3f2bf7f3e9850107821c93cc357b5139 |
