# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22f847bbdb3bff7bf5823cbe39964b330b3ee1ba23484549f7f4546fac1981a9`
- fixture hash: `sha256-a7f2ea82d1ad7a3badc44ebc7ebcd547c985d36abe3fcd06170981ec576de057`
- score hash: `sha256-14aed2475058cc31777b3644109d928b4c2640252a61b70f4a2e78450eb351b9`
- bundle hash: `sha256-e58cbcaf220769f5302f771461032a69a4da84667962254e1ced7290820ea082`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9025e023a7eb98100239409dc6df273a8fbdc8529118429bd0cb2b4995877ef2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a1ee2a144fe130807ac12c308ee664c3691efd737d6083d2048775fcc01c3925 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e58517d8868e76ff5f91b35bfb7aa6d63a11b290f36caa779fc22b9d4346919e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-29508fd9efb33ec2114e8afa70378b5b9c99eaa0f853cd4d6f760e570a87c8d5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-88e06a94 | sha256-7fa6613f58f0af9ae0cdd661948133aaca138095cf91d7b25abe6e035a0ab676 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-88e06a94 | sha256-1ecaddeb77cda0514c79e6f81457b57730f3673338c8a98d1e8c79915439fc1d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-34ea8861 | sha256-2811f89510d0af64c5e9b993214c00a4f5bb87c7fbaf8498ac26e14179233ffe |
