# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-052`
- winner mode: `graph_prior_only`
- trace hash: `sha256-eb4ce4c16a0b4086f9bf16153627317bccc66c138e3f3eabb740de5aad356d3c`
- fixture hash: `sha256-f8df6b8b0d3896e4d68df7e66273fb59a221dba4842848c8bd3431e1201171eb`
- score hash: `sha256-d53728b8b82a69e753d7db450b189d35a692a0f5d110b34c6f9f10a444c30074`
- bundle hash: `sha256-42b89cdb71fa71eced0b20b95c1605f898f78b8fe11ceb2ebbed825feb6367ab`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-31a831566ed58685dea8a0c35a91e51999c06d52779d7820057deddb5dbf99cd |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-23fbf2e012ee56cc3fdbe8f25eb86c2972996f3a3f03e24535daf9947e6c2b00 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ad1290afdc32e83a503aa5e9d40469e5eb55ecffcbbda53fc5611fe152724364 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-477d6da2f19181ca7f5e1861aea9d61f5c035ea7d169c186585ecbc91394300e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d3352b91 | sha256-ab6cfdcda9d8e6fcf3f822dd13639b80e1a4e70213f8c58b2a62d3f9995f58ec |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-d3352b91 | sha256-596efeec1bdafe8af1435eaa35a7e243891651128be035c3c90708208a985744 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-6a5a693c | sha256-764d3348b29af542003636968fb23ef3581b21200b6145148c8c9d183e427d45 |
