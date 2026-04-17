# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70ab1ae3977bf5b8f105672a2af7f511f5d5e8eab54227af0f4c11c32810b91e`
- fixture hash: `sha256-f43d8483c3b4eb473890c9d4aad38b8eb4a81081d719d9c58fd2752db7997c33`
- score hash: `sha256-7c89560d337ae2939bda0f98d86912650d84c64b5bd8ad099d513ee56feec6ec`
- bundle hash: `sha256-00f60ea0f4314e1d0acb8ea7a530236ccbf5d5e3cd816052cdf5c8edf610e5dc`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0af8593686f4dcd1625a4737259415fed87f48af0fee073ee2e87cde2bfd51e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8d77dbad9004ee740ca6b01d10edda8b4f06318f85f6526b3a9e36ad198450fd |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86cba97c83f4859fcc4b05ec1d3b4a1e0741e31884a56e1185970393a2d020e5 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-adaf0371080f9045bb0481b12e69daa13e80b59db80fde2de4fa4e7c81a09c91 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2a322c62 | sha256-9468b8c3f6bfb0f041d51ef0297d8e977f326049db18d6a9680b014af56c9a73 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-2a322c62 | sha256-b4cc2c5a06781d34f19d00072bfb609186314fdb730b35d5e12f32530be82345 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-561c64d1 | sha256-1f9ed7b4fbc203bdf6fd792a1e01265cf4f8db62b9a8191adb20c58b00aba8e5 |
