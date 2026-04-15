# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-22f847bbdb3bff7bf5823cbe39964b330b3ee1ba23484549f7f4546fac1981a9`
- fixture hash: `sha256-a7f2ea82d1ad7a3badc44ebc7ebcd547c985d36abe3fcd06170981ec576de057`
- score hash: `sha256-c39c5dd96c3731f6a0823272b3d3ddc09425222bb6a59489c6798239bc26601a`
- bundle hash: `sha256-5ec438f56dfd6bb23428e62998d09cac2e00b4f55b8337c6e253468222ce4815`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9025e023a7eb98100239409dc6df273a8fbdc8529118429bd0cb2b4995877ef2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a6996a1bcec4dc1a9062504d196cf68e6e2bb8c268ef130ac7f9a715541b8072 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d79755f63348dcd89d59502d602a745c9bd761028573011a0d244a18d13e3767 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-8e520146d3e4efb22302bdbc2f7ad87a915352cadd52fe1bb733bdbe0e3ebd32 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27251736 | sha256-5391aa73f2dd489296b2a5cdd52b16653da2a3edf5d0d92c3f6aad6843774450 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-27251736 | sha256-918287ae42762ff7c4a555d4ba7820ac94ac4ed7a02368dfc4fe158b4572500a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-27251736 | sha256-5391aa73f2dd489296b2a5cdd52b16653da2a3edf5d0d92c3f6aad6843774450 |
