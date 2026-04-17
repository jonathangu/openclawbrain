# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-c77161fd107da0850cb368a9be7a432917f75dca7a822871991fd4fdb28ea1b9`
- fixture hash: `sha256-305a6c8327f5df890119bbc3711133fb545cdd219a8e22832dd1a9b40c670ed7`
- score hash: `sha256-d4d44491467d011e0eed5ea113dcb9c74297e1cc1a37453b3387e6ce0ef38f51`
- bundle hash: `sha256-ae30bfc126f80f8d37e63eddf14d31d0f7e814fa96878ca532f24e57c6bdf9cb`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9994dc36549b50fc869a0d07853fce421c050985a49a6ae0b76d1bc12cb356c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9bc79df8aae73397c49209d2259c9926b275df7a29c620a8cd39835a9c7d5541 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6feecaf98787a3065333903c743b9cf9348c1213d7b3ad130da3b657bbc4ec08 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-efabc7cf8f89875ffbf949103a1a0d14e21b6373044fb14c211ae94789de900d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c85b44f8 | sha256-46de7471e559fe2f41aafa4e2c07b1463e7e4f0bc46556d66c9921f1fb71bb6c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c85b44f8 | sha256-483b12979ed293c4c0fa4ef22a1b8952d973a2860102111bca300bca695f74fc |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-15c1883f | sha256-4f95aa9f97de19fbdf39d600cb92b9b83bc91e82b4f956aa7a909cb33750984e |
