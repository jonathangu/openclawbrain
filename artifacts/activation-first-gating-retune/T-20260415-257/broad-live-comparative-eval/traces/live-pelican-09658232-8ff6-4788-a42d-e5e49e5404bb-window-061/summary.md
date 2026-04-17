# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-061`
- winner mode: `graph_prior_only`
- trace hash: `sha256-70ab1ae3977bf5b8f105672a2af7f511f5d5e8eab54227af0f4c11c32810b91e`
- fixture hash: `sha256-f43d8483c3b4eb473890c9d4aad38b8eb4a81081d719d9c58fd2752db7997c33`
- score hash: `sha256-500e5be4d6081430a35fca2014a463609eeb2c4b3d2c7b84bbcca7768471ac18`
- bundle hash: `sha256-81767e78dc8de5ba5d748d36c10af4b81cba3a3c1f62680d322d96d141d17b04`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b0af8593686f4dcd1625a4737259415fed87f48af0fee073ee2e87cde2bfd51e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-34578b3fa06183bf6655ee45ae62f11ceb8e79154316810291be203d0d993095 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-08ce597b14ef4450e2ea3c9f935efe6292b5c38194cb648439dbfa2f7fdae474 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-ed566a800dfcb5000a2708e31c647a8e2cca2c876fcd1d70b32784a6e94e6a07 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cad4ca17 | sha256-ea8fdc4155561a25b4f0e0dc6a8aa6a3574d6e36e9133ea452e1cb381345b231 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-cad4ca17 | sha256-07ad84e9577bbfa785c523774572f77951eac2d3ab63d59d7a3c94134fa225c5 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-f6bf0286 | sha256-2c23780fbb32ecf94bff4828081113158370f548b3b8c6774ba01e020fc82b51 |
