# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eb4ce4c16a0b4086f9bf16153627317bccc66c138e3f3eabb740de5aad356d3c`
- fixture hash: `sha256-f8df6b8b0d3896e4d68df7e66273fb59a221dba4842848c8bd3431e1201171eb`
- score hash: `sha256-6236c86d85e8eacc1f127b1f82b30d0d83e011756c191982b5e81f9cc265b26d`
- bundle hash: `sha256-33e624bfe4a2d786aecbc6305825da6c3489e7ef69b0ff23af3cde8cf06e2eb6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31a831566ed58685dea8a0c35a91e51999c06d52779d7820057deddb5dbf99cd |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5ecbf368e9ae7335196f82f3d8dbb804799898485f984a038fc031dcf42beb0e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-98f791e303a5514a55267bac13becee8bcb21bff75450f3edb4e9f10de9374cc |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-21648272ee0f5253e62de90175a639705e375f5b6ba264734c445131b23e9a6b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa402dba | sha256-fd537d3aee7b2b08db7d6527a9f6fd14b2d99a7d21a33dc49640305090e47e53 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-aa402dba | sha256-a668c20b4738ff8abcfb47e018413009dc1c756c3100ee8e586711d965546a3c |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-41656b65 | sha256-d7143ca434df3d5fcb43fcb0031118b5fcbeb359cb1e113813de5c3edbd415ec |
