# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-6c49ca2fe441629d93695e938073dd41facc650cc9fd301e1fa807efab482f72`
- fixture hash: `sha256-a51ac08634e3c4803c3d3973ce1a7c858ffb1429844452de0ed6e3279b36730b`
- score hash: `sha256-f7e512d8f85f5550b31dcd591ea9cdebb1fb8341fefe7bb695dec3b8cf066d7c`
- bundle hash: `sha256-bdbf01c4e5c5fdf72b40de6c9357e336d0f00e5900204d5b89b2578339dee720`

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
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a6c2e3cc9fd6ee1604d2c526310b1ddca47a11a50e7f9573ba696d4001f01dac |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-812b7419d47bad5e5e2b25f927e4ef093adf028c97466cf6832c82b4080ae035 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c0e7040c304d8d80b08ebc62a76ef1d1846b9b74a58949e4c4a7c32c909968c9 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-ade45293091725e91404b55ba7b58a1d9705ba1d7c2e3c771ab4eb4b24aeff3d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1b8043f3 | sha256-acb5c203b8e372da092f5e04d6fbad7269b267269fcdcb743bdafbdb5bd56118 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1b8043f3 | sha256-0d2b8e15d84b10002db9d50832ae7ebb8107e5b7044e0027219a07512f305dfc |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-9ef01a0c | sha256-6d6defbab693e3ff4f92655a51ec406f192ae047f34a78f19040cfcc237352c1 |
