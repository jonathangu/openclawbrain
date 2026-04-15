# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c77161fd107da0850cb368a9be7a432917f75dca7a822871991fd4fdb28ea1b9`
- fixture hash: `sha256-305a6c8327f5df890119bbc3711133fb545cdd219a8e22832dd1a9b40c670ed7`
- score hash: `sha256-536dfcd009bba4d265fe5a42a1345421557456f31aee224c57ddc8b7436be02f`
- bundle hash: `sha256-049d22a1f8b919602cc6ee39c73f7d72f0f132090d83663e35f91bd6e4d54141`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9994dc36549b50fc869a0d07853fce421c050985a49a6ae0b76d1bc12cb356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f02a9dceb5046d6e1dc90d9431d9cc06b8de54b9ae837c2250bc261f28db7d63 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee8abc4b6550f6313baabd57e0e2f27fd9b7d438323adb3e0f008dc02050660a |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-ae1b9ecd220b18129eea985a4fb2a51c3c56399b4610a656926e206c031a74b0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-95ea2779 | sha256-a4cfef70628b4ef1af902ec650c10c51c5eccbadc30848fcb514d40ccebe7f71 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-95ea2779 | sha256-1bbc0d61a6c8625101f96457f7c5c339aa625a25871e2b327a5665464a827cb8 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-95ea2779 | sha256-a4cfef70628b4ef1af902ec650c10c51c5eccbadc30848fcb514d40ccebe7f71 |
