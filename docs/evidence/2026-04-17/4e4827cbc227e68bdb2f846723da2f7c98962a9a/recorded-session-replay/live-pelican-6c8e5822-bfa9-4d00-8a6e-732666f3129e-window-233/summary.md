# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-233`
- winner mode: `graph_prior_only`
- trace hash: `sha256-9d936165695614f904d36571a8a48065c182dddc8afd06f7b5a7de26e3d1a3da`
- fixture hash: `sha256-6ad09120c53334c8df0b9f19b852f07c2aa8ca071680e8461d1d0fad693137b2`
- score hash: `sha256-d660076ce111dbdb44ba4806ce37631f5e262dccbf7b88b13b80e4e1aca2e81f`
- bundle hash: `sha256-8b232d6b03a421b8c730463a81be07751778113e257be2e74feb4ebf4ad48761`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3329ac350a9048e47f1760a5c97b317667c0cdc04bb3d7fb2085cb6158792e13 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c286050e191c57a0dec284c2a9bb471c2d3be18d24a3c26e7ea6ff6fe3ab077d |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-800485979bf9562f7e5a985643dc2ae62101012cc301b6d8e6b27b6ee6e75b7e |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-721882c549daa915aec2e6c1119ae217b039cfc00374d1ca679be9cee7065885 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-56be2438 | sha256-57f1c5b44e21ea8a6210dd8d69e23ab5763769e650cec46f312994e3abbc5db8 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-56be2438 | sha256-f18d18a2bb27fffda739954b11b48eeb382a8f60f0d1ccaf55968a9b23495c06 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-c909f563 | sha256-71550a5c7e8213d3d9b3adbfacd4bf78f780ef078c8ff7f1087f8c0a76a9690c |
