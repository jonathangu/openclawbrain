# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-945c2e59668de577944ec7fc5dd5f9442c630538679d6020f6fdac64e2a21a17`
- fixture hash: `sha256-19b533ee2cadb7bef94e2f868a3d98284f247e98f26920f7fea15136681e3d11`
- score hash: `sha256-2cf0e1da0d11ceea61ea3cbe173dc6fafdcac4fac64c86db3bffbc825e52b212`
- bundle hash: `sha256-77e8f6a72d5b8c542bec061b5222e1a0bdefae833aa825286d9607c06ae65f2e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d37994d9aa92d2e1c7fd5cea54b3093f268f662f580c5608c088fa86597acbc2 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cde85ba766983da3b5e10633aaab10ba61ac64c56bd88a1e1be399ae65619a2e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-24d9448c81690a84b7cc9b9ceb2b298ff0d0af1817c6e7434ed3645287ada4fb |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-ed3a71087b4a8808d9f91628b6fea107bef3087a27ead39d8df838d63c8546c2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-361bcb7a | sha256-3827cf0e5ef5fe82825f932ca91919d9a5af26ee8c940881cdc1d647ba0579ad |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-361bcb7a | sha256-0206626e184e7de97cbfcc2f5272bf28ef908db563743d084dbba1d00ce0645e |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-fd1bb507 | sha256-0e100ed44f8181b485fbab2a70194c25b85ba03cc061d54fc74fc99fbc572b1d |
